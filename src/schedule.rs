use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::str::FromStr;

#[derive(Debug)]
pub struct ScheduleInput {
    pub days_in_week: u32,
    pub periods_in_day: u32,
    pub classes: Vec<Class>,
    pub teachers: Vec<Teacher>,
    pub rooms: Vec<Room>,
    pub unused_periods: usize,
}

impl ScheduleInput {
    pub fn new(days_in_week: u32, periods_in_day: u32) -> Self {
        ScheduleInput {
            days_in_week,
            periods_in_day,
            classes: Vec::new(),
            teachers: Vec::new(),
            rooms: Vec::new(),
            unused_periods: 0,
        }
    }

    pub fn update(&mut self) {
        self.classes.sort_by(|a, b| match b.size.cmp(&a.size) {
            Ordering::Equal => b.meetings_per_week.cmp(&a.meetings_per_week),
            x @ _ => x,
        });
        self.rooms.sort_by(|a, b| b.capacity.cmp(&a.capacity));

        self.unused_periods = self.periods_in_day as usize * self.rooms.len() - self.classes.len();
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Class {
    pub name: String,
    pub size: u32,
    pub meetings_per_week: u32,
}

impl Debug for Class {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {})",
            self.name, self.size, self.meetings_per_week
        )
    }
}

impl FromStr for Class {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut split = s.split(":");

        let name = match split.next() {
            Some(name) => name.trim().to_owned(),
            None => return Err("expected class name"),
        };

        let size = match split.next() {
            Some(size) => size
                .trim()
                .parse::<u32>()
                .map_err(|_| "failed to parse class size integer")?,
            None => return Err("expected class size after the colon"),
        };

        let meetings_per_week = match split.next() {
            Some(size) => size
                .trim()
                .parse::<u32>()
                .map_err(|_| "failed to parse class metting count integer")?,
            None => return Err("expected class meeting count after the second colon"),
        };

        Ok(Class {
            name,
            size,
            meetings_per_week,
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Teacher {
    pub name: String,
    pub class_preferences: Vec<String>,
    pub period_preferences: Vec<u32>,
}

impl Debug for Teacher {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?}", self.name, self.class_preferences)
    }
}

impl FromStr for Teacher {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut split = s.split(":");

        let name = match split.next() {
            Some(name) => name.trim().to_owned(),
            None => return Err("expected teacher name"),
        };

        let period_preferences = match split.next() {
            Some(periods) => {
                let mut period_preferences = Vec::new();
                for period in periods.split(',').map(|s| {
                    s.trim()
                        .parse::<u32>()
                        .map_err(|_| "failed to parse perferred period")
                }) {
                    period_preferences.push(period?);
                }
                period_preferences
            }
            None => Vec::new(),
        };

        Ok(Teacher {
            name,
            class_preferences: Vec::new(),
            period_preferences,
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Room {
    pub name: String,
    pub capacity: u32,
    pub days_available: Vec<u32>,
}

impl Debug for Room {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) {:?}",
            self.name, self.capacity, self.days_available
        )
    }
}

impl FromStr for Room {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut split = s.split(":");

        let name = match split.next() {
            Some(name) => name.trim().to_owned(),
            None => return Err("expected room name"),
        };

        let capacity = match split.next() {
            Some(capacity) => capacity
                .trim()
                .parse::<u32>()
                .map_err(|_| "failed to parse room capacity integer")?,
            None => return Err("expected room capacity after the colon"),
        };

        Ok(Room {
            name,
            capacity,
            days_available: Vec::new(),
        })
    }
}
