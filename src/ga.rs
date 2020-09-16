use crate::schedule::{Class, Room, ScheduleInput, Teacher};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use std::cmp;
use std::collections::HashMap;
use std::convert::{AsMut, AsRef};
use std::fmt::{self, Display, Formatter};
use std::iter::*;
use std::mem;

pub struct Settings {
    pub reproduce_percent: f32,
    pub elitist_percent: f32,
    pub immigration_percent: f32,
    pub mutate_prob: f32,
}

pub struct Schedule(Vec<Vec<HashMap<Room, (Class, Teacher)>>>);

impl Schedule {
    pub fn new(input: &ScheduleInput) -> Self {
        Schedule(vec![
            vec![HashMap::new(); input.periods_in_day as usize];
            input.days_in_week as usize
        ])
    }

    pub fn get(&self, day: usize, period: usize) -> &HashMap<Room, (Class, Teacher)> {
        &self.0[day - 1][period - 1]
    }

    pub fn add(&mut self, day: usize, period: usize, room: Room, class: Class, teacher: Teacher) {
        self.0[day - 1][period - 1].insert(room, (class, teacher));
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut flag = false;
        for (day, periods) in self.0.iter().enumerate() {
            write!(f, "{}Day: {}", if flag { "\n" } else { "" }, day + 1)?;
            flag = true;
            for (period, classes) in periods.iter().enumerate() {
                write!(
                    f,
                    "\n#{}: {}",
                    period + 1,
                    classes
                        .iter()
                        .map(|(room, (class, teacher))| format!(
                            "{}:{} @{}",
                            class.name, teacher.name, room.name
                        ))
                        .join(", ")
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct Individual {
    pub map_class_teacher: OwnedChromosome,
    pub class_ordering: OwnedChromosome,
    pub class_day_selections: Vec<OwnedChromosome>,
    pub buffer_ordering: OwnedChromosome,
    last_loss: f32,
    needs_loss_update: bool,
}

impl Individual {
    pub fn new(input: &ScheduleInput) -> Self {
        let mut class_day_selections = Vec::with_capacity(input.classes.len());
        class_day_selections.resize_with(class_day_selections.capacity(), || {
            Chromosome::new(input.days_in_week as usize)
        });

        Individual {
            map_class_teacher: Chromosome::new(input.classes.len()),
            class_ordering: Chromosome::new(input.classes.len()),
            class_day_selections,
            buffer_ordering: Chromosome::new(input.unused_periods),
            last_loss: 0.0,
            needs_loss_update: true,
        }
    }

    pub fn evaluate_loss(&mut self, input: &ScheduleInput) -> f32 {
        if !self.needs_loss_update {
            self.last_loss
        } else {
            self.evaluate(input).1
        }
    }

    pub fn evaluate(&mut self, input: &ScheduleInput) -> (Schedule, f32) {
        let mut schedule = Schedule::new(input);

        let period_count = input.periods_in_day as usize;
        let mut day_buffer: Vec<u32> = Vec::with_capacity(input.days_in_week as usize);

        let classes_with_teachers = self
            .map_class_teacher
            .as_mapping()
            .map(|(class_index, teacher_index)| {
                (
                    &input.classes[class_index],
                    &input.teachers[(input.teachers.len() * teacher_index) / input.classes.len()],
                )
            })
            .collect::<Vec<_>>();

        let mut rooms_mut = input
            .rooms
            .iter()
            .flat_map(|room| repeat(room).take(input.periods_in_day as usize))
            .cloned()
            .collect::<Vec<_>>();

        let mut class_anchor: usize = 0;
        let mut room_anchor: usize = 0;
        while class_anchor < input.classes.len() && room_anchor < rooms_mut.len() {
            let mut ct_buffer = [0usize; 8];
            ct_buffer[0] = class_anchor;
            let mut class_index = class_anchor;
            let mut room_index = room_anchor + 1;

            while room_index < rooms_mut.len()
                && rooms_mut[room_index].capacity >= input.classes[class_anchor].size
            {
                room_index += 1;
            }

            let min_capacity = rooms_mut
                .get(room_index)
                .map(|room| room.capacity)
                .unwrap_or(0);
            while class_index < input.classes.len()
                && min_capacity <= input.classes[class_index].size
            {
                class_index += 1;
            }

            let mut i = 1;
            let mut last_mpw = classes_with_teachers[class_anchor].0.meetings_per_week;
            for (index, class_and_teacher) in classes_with_teachers[class_anchor..class_index]
                .iter()
                .enumerate()
            {
                if class_and_teacher.0.meetings_per_week != last_mpw {
                    last_mpw = class_and_teacher.0.meetings_per_week;
                    ct_buffer[i] = class_anchor + index;
                    i += 1;
                }
            }
            ct_buffer[i] = class_index;

            for (class_set, window) in ct_buffer[0..=i]
                .windows(2)
                .map(|window| (&classes_with_teachers[window[0]..window[1]], window))
            {
                let required_slots = class_set[0].0.meetings_per_week as usize;
                let mut available_rooms: Vec<Option<(usize, &mut Room)>> = rooms_mut
                    [room_anchor..room_index]
                    .iter_mut()
                    .enumerate()
                    .map(|(index, room)| ((index + room_anchor) % period_count + 1, room))
                    .filter(|(_, room)| room.days_available.len() >= required_slots)
                    .map(|x| Some(x))
                    .collect::<Vec<_>>();
                let mut room_buffer = Vec::new();

                let class_chromosome = self.class_ordering.substring(window[0], window[1]);
                let buffer_chromosome = if class_set.len() < available_rooms.len() {
                    self.buffer_ordering
                        .substring(0, available_rooms.len() - class_set.len())
                } else {
                    self.buffer_ordering.substring(0, 0)
                };

                for (_, room_index) in
                    Chromosome::joint_mapping(&class_chromosome, &buffer_chromosome)
                        .take(class_set.len())
                {
                    room_buffer.push(available_rooms[room_index].take().unwrap());
                }

                'class_assignment: for (index, &(class, teacher)) in class_set.iter().enumerate() {
                    for (period, room) in room_buffer.iter_mut() {
                        for (_, day_index) in self.class_day_selections[window[0] + index]
                            .substring(0, room.days_available.len())
                            .as_mapping()
                        {
                            day_buffer.push(room.days_available[day_index])
                        }

                        day_buffer.drain_filter(|&mut day| {
                            schedule
                                .get(day as usize, *period)
                                .values()
                                .any(|(_, Teacher { name, .. })| name == &teacher.name)
                        });

                        if day_buffer.len() >= required_slots {
                            for day in day_buffer.iter().cloned().take(required_slots) {
                                schedule.add(
                                    day as usize,
                                    *period,
                                    room.clone(),
                                    class.clone(),
                                    teacher.clone(),
                                );
                                let idx =
                                    room.days_available.iter().position(|x| *x == day).unwrap();
                                room.days_available.remove(idx);
                            }

                            day_buffer.clear();
                            continue 'class_assignment;
                        }

                        day_buffer.clear();
                    }
                }
            }

            class_anchor = class_index;
            room_anchor = room_index;
        }

        let class_preference_satisfaction = classes_with_teachers
            .iter()
            .map(|(class, teacher)| {
                teacher
                    .class_preferences
                    .iter()
                    .position(|preference| class.name.starts_with(preference))
                    .unwrap_or(teacher.class_preferences.len()) as f32
            })
            .sum::<f32>()
            / classes_with_teachers.len() as f32;

        let total_class_count = input
            .classes
            .iter()
            .map(|class| class.meetings_per_week)
            .sum::<u32>() as f32;
        let period_satisfaction = schedule
            .0
            .iter()
            .flat_map(|periods| periods.iter().enumerate())
            .map(|(period, classes)| {
                classes
                    .iter()
                    .map(|(_, (_, teacher))| {
                        if teacher.period_preferences.contains(&(period as u32 + 1))
                            || teacher.period_preferences.is_empty()
                        {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / total_class_count;

        self.last_loss = period_satisfaction;

        self.needs_loss_update = false;
        (schedule, self.last_loss)
    }

    pub fn recombine_all<R: Recombinator>(
        first: &mut Self,
        second: &mut Self,
        recombinator: &R,
        rng: &mut impl Rng,
    ) {
        first.needs_loss_update = true;
        second.needs_loss_update = true;

        recombinator.recombine(
            &mut first.map_class_teacher,
            &mut second.map_class_teacher,
            rng,
        );

        recombinator.recombine(&mut first.class_ordering, &mut second.class_ordering, rng);
        recombinator.recombine(&mut first.buffer_ordering, &mut second.buffer_ordering, rng);

        for (first_day_selection, second_day_selection) in first
            .class_day_selections
            .iter_mut()
            .zip(second.class_day_selections.iter_mut())
        {
            recombinator.recombine(first_day_selection, second_day_selection, rng);
        }
    }

    pub fn mutate_all(&mut self, probability: f32, rng: &mut impl Rng) {
        macro_rules! mutate {
            ($chromo:expr) => {
                if rng.gen::<f32>() < probability {
                    $chromo.point_mutation(rng.gen::<usize>() % $chromo.len(), rng);
                    self.needs_loss_update = true;
                }
            };
        }

        mutate!(self.map_class_teacher);
        mutate!(self.class_ordering);
        mutate!(self.buffer_ordering);

        for day_selection in self.class_day_selections.iter_mut() {
            mutate!(day_selection);
        }
    }
}

#[inline]
pub fn slice_crossover<T>(first: &mut [T], second: &mut [T], start: usize, end: usize) {
    (&mut first[start..end]).swap_with_slice(&mut second[start..end]);
}

#[derive(Debug, Clone)]
pub struct Chromosome<T>(T);

pub type OwnedChromosome = Chromosome<Vec<f32>>;

impl Chromosome<Vec<f32>> {
    pub fn new(len: usize) -> Self {
        Chromosome((0..len).map(|_| thread_rng().gen()).collect())
    }
}

impl<T: AsRef<[f32]>> Chromosome<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.as_ref().len()
    }

    pub fn substring(&self, start: usize, end: usize) -> Chromosome<&'_ [f32]> {
        Chromosome(&self.0.as_ref()[start..end])
    }

    pub fn as_mapping(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        Self::mapping_internal(self.0.as_ref().iter())
    }

    pub fn joint_mapping<'a, 'b: 'a>(
        first: &'a Self,
        second: &'b Self,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        Self::mapping_internal(first.as_ref().iter().chain(second.as_ref()))
    }

    fn mapping_internal<'a>(
        iter: impl Iterator<Item = &'a f32>,
    ) -> impl Iterator<Item = (usize, usize)> {
        iter.copied()
            .enumerate()
            .sorted_by(|(_, key1), (_, key2)| {
                key1.partial_cmp(key2).unwrap_or(cmp::Ordering::Equal)
            })
            .enumerate()
            .map(|(original_index, (mapped_index, _))| (original_index, mapped_index))
    }
}

impl<T: AsMut<[f32]>> Chromosome<T> {
    pub fn point_mutation(&mut self, index: usize, rng: &mut impl Rng) {
        self.0.as_mut()[index] = rng.gen();
    }
}

impl<T: AsRef<[f32]>> AsRef<[f32]> for Chromosome<T> {
    fn as_ref(&self) -> &[f32] {
        self.0.as_ref()
    }
}

impl<T: AsMut<[f32]>> AsMut<[f32]> for Chromosome<T> {
    fn as_mut(&mut self) -> &mut [f32] {
        self.0.as_mut()
    }
}

pub trait Recombinator {
    fn recombine<T>(
        &self,
        first: &mut Chromosome<T>,
        second: &mut Chromosome<T>,
        rng: &mut impl Rng,
    ) where
        T: AsRef<[f32]> + AsMut<[f32]>;
}

pub struct KPoint {
    count: f32,
}

impl KPoint {
    pub fn new(count: usize) -> Self {
        assert!(count > 0);

        KPoint {
            count: (count - 1) as f32,
        }
    }

    fn single_point<T>(first: &mut Chromosome<T>, second: &mut Chromosome<T>, rng: &mut impl Rng)
    where
        T: AsRef<[f32]> + AsMut<[f32]>,
    {
        let cut = (rng.gen::<f32>() * first.len() as f32) as usize;
        let len = first.len();
        slice_crossover(first.as_mut(), second.as_mut(), cut, len);
    }
}

impl Recombinator for KPoint {
    fn recombine<T>(
        &self,
        first: &mut Chromosome<T>,
        second: &mut Chromosome<T>,
        rng: &mut impl Rng,
    ) where
        T: AsRef<[f32]> + AsMut<[f32]>,
    {
        assert_eq!(
            first.len(),
            second.len(),
            "Cannot recombine chromosomes of different lengths"
        );

        if self.count == 0.0 {
            Self::single_point(first, second, rng);
            return;
        }

        let step = (first.len() as f32) / (self.count + 1.0);
        let mut i = 0.0f32;
        let mut start: f32;
        let mut end = rng.gen::<f32>() * step;
        while i < self.count {
            i += 1.0;
            start = end;
            end = step * (rng.gen::<f32>() + i);
            slice_crossover(
                first.as_mut(),
                second.as_mut(),
                start as usize,
                end.round() as usize,
            );
        }
    }
}

pub struct Uniform {
    weight: f32,
}

impl Uniform {
    pub fn new() -> Self {
        Uniform { weight: 0.5 }
    }

    pub fn weighted(weight: f32) -> Self {
        Uniform { weight }
    }
}

impl Recombinator for Uniform {
    fn recombine<T>(
        &self,
        first: &mut Chromosome<T>,
        second: &mut Chromosome<T>,
        rng: &mut impl Rng,
    ) where
        T: AsRef<[f32]> + AsMut<[f32]>,
    {
        assert_eq!(
            first.len(),
            second.len(),
            "Cannot recombine chromosomes of different lengths"
        );

        for i in 0..first.len() {
            if rng.gen::<f32>() < self.weight {
                mem::swap(&mut first.as_mut()[i], &mut second.as_mut()[i]);
            }
        }
    }
}

pub struct RouletteWheelSelection {
    input: ScheduleInput,
    loss_buffer: Vec<f32>,
}

impl RouletteWheelSelection {
    pub fn new(mut input: ScheduleInput) -> Self {
        input.update();

        RouletteWheelSelection {
            input,
            loss_buffer: Vec::new(),
        }
    }

    pub fn evolve<R>(
        &mut self,
        settings: &Settings,
        population: &mut [Individual],
        dest_population: &mut Vec<Individual>,
        recombinator: &R,
    ) -> f32
    where
        R: Recombinator,
    {
        let mut rng = thread_rng();
        dest_population.clear();

        // Compute the loss vector
        let n = population.len();
        self.loss_buffer.resize_with(n, || 0.0);
        let mut loss_sum: f32 = 0.0;
        let mut min_loss = f32::MAX;
        for i in 0..n {
            let loss = population[i].evaluate_loss(&self.input);
            self.loss_buffer[i] = loss;
            loss_sum += loss;

            if loss < min_loss {
                min_loss = loss;
            }
        }
        self.loss_buffer
            .iter_mut()
            .for_each(|loss| *loss /= loss_sum);
        self.loss_buffer
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));
        population.sort_by(|a, b| {
            a.last_loss
                .partial_cmp(&b.last_loss)
                .unwrap_or(cmp::Ordering::Equal)
        });

        // Copy elites
        let elite_count = (settings.elitist_percent * population.len() as f32) as usize;
        (0..elite_count)
            .map(|index| population[index].clone())
            .for_each(|elite| dest_population.push(elite));

        // Add immigrants
        let immigrant_count = (settings.immigration_percent * population.len() as f32) as usize;
        for _ in 0..immigrant_count {
            dest_population.push(Individual::new(&self.input));
        }

        // Generate offspring
        while dest_population.len() < population.len() {
            // Get the two parents
            let mut selections = [0usize; 2];
            'selector: for i in 0..2 {
                let mut random = rng.gen::<f32>();
                for j in 0..n {
                    if random < self.loss_buffer[j] && (i == 0 || selections[0] != j) {
                        selections[i] = j;
                        continue 'selector;
                    }

                    random -= self.loss_buffer[j];
                }

                selections[i] = n - 1;
            }

            // Compute the child chromosomes
            let mut first = population[selections[0]].clone();
            let mut second = population[selections[1]].clone();
            Individual::recombine_all(&mut first, &mut second, recombinator, &mut rng);
            first.mutate_all(settings.mutate_prob, &mut rng);
            second.mutate_all(settings.mutate_prob, &mut rng);

            // Update minimum loss value
            let loss = first.evaluate_loss(&self.input);
            if loss < min_loss {
                min_loss = loss;
            }
            let loss = second.evaluate_loss(&self.input);
            if loss < min_loss {
                min_loss = loss;
            }

            // Add them to the population
            dest_population.push(first);

            if dest_population.len() < population.len() {
                dest_population.push(second);
            } else {
                break;
            }
        }

        min_loss
    }
}
