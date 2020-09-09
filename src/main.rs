mod ga;
mod schedule;

use ga::*;
use schedule::*;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::str::FromStr;

const POPULATION_SIZE: usize = 100;
const MAX_ITERATIONS: usize = 1000;
const SETTINGS: Settings = Settings {
    reproduce_percent: 0.7,
    elitist_percent: 0.2,
    immigration_percent: 0.1,
    crossover_prob: 1.0,
    mutate_prob: 0.01,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Open input.txt for the test cases
    let input_file = match File::open("input.txt") {
        Ok(file) => file,
        Err(_) => {
            println!("Expected input file input.txt in current working directory.");
            return Ok(());
        }
    };

    // Read the file line-by-line
    let mut lines = BufReader::new(input_file).lines().peekable();

    let days_in_week: u32 =
        next_line(&mut lines)?.expect("Expected number of days in the scholastic week.");
    let periods_in_day: u32 =
        next_line(&mut lines)?.expect("Expected the number of periods in the scholastic day.");

    let mut input = ScheduleInput::new(days_in_week, periods_in_day);

    let class_count: usize = next_line(&mut lines)?.expect("Expected the number of classes");
    for _ in 0..class_count {
        let class: Class =
            next_line(&mut lines)?.expect("Expected a class in the form \"name: size\"");
        input.classes.push(class);
    }

    let teacher_count: usize = next_line(&mut lines)?.expect("Expected the number of teachers");
    for _ in 0..teacher_count {
        let mut teacher: Teacher = next_line(&mut lines)?.expect("Expected a teacher\'s name");

        // Read preferences
        while let Some(Ok(line)) = lines.peek() {
            if !line.trim().starts_with("-") {
                break;
            }

            let preference = lines.next().unwrap().unwrap().trim()[1..].trim().to_owned();
            teacher.preferences.push(preference);
        }

        input.teachers.push(teacher);
    }

    let room_count: usize = next_line(&mut lines)?.expect("Expected the number of rooms");
    for _ in 0..room_count {
        let mut room: Room =
            next_line(&mut lines)?.expect("Expected a room in the form \"name: capacity\"");
        room.days_available = (1..=days_in_week).collect();

        // Read unavailable days
        while let Some(Ok(line)) = lines.peek() {
            if !line.trim().starts_with("-") {
                break;
            }

            let day: u32 = lines.next().unwrap().unwrap().trim()[1..].trim().parse()?;
            let index = room.days_available.iter().position(|x| *x == day).unwrap();
            room.days_available.remove(index);
        }

        input.rooms.push(room);
    }

    // TODO: remove
    input.update();
    let mut population = Vec::with_capacity(POPULATION_SIZE);
    population.resize_with(population.capacity(), || Individual::new(&input));
    population[0].evaluate(&input);
    let mut aux_population = population.clone();
    let recombinator = Uniform::weighted(0.3);
    let mut selector = RouletteWheelSelection::new(input);

    Ok(())
}

fn next_line<T>(
    lines: &mut impl Iterator<Item = Result<String, io::Error>>,
) -> Result<Option<T>, Box<dyn Error>>
where
    T: FromStr,
    T::Err: Into<Box<dyn Error>>,
{
    while let Some(line) = lines.next() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            return Ok(Some(T::from_str(trimmed).map_err(Into::into)?));
        }
    }

    Ok(None)
}
