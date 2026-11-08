use chrono::{DateTime, Datelike, Duration, Month, NaiveDate, TimeZone, Timelike, Utc, Weekday};
use std::ops::{Add, Sub};

#[derive(Debug)]
pub enum WeekStartDay {
    Monday,
    Sunday,
}

impl WeekStartDay {
    pub fn to_week_day(&self) -> Weekday {
        match self {
            WeekStartDay::Monday => Weekday::Mon,
            WeekStartDay::Sunday => Weekday::Sun,
        }
    }
}

pub trait TimeZoneNow {
    type Timezone: TimeZone;
    fn now(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_minute(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_hour(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_day(&self) -> DateTime<Self::Timezone>;
    /// get beginning of week. the default week start day is Monday.
    fn beginning_of_week(&self) -> DateTime<Self::Timezone>;
    /// get beginning of week given specific week start day.
    fn beginning_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone>;

    fn beginning_of_month(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_quarter(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_year(&self) -> DateTime<Self::Timezone>;

    fn end_of_minute(&self) -> DateTime<Self::Timezone>;
    fn end_of_hour(&self) -> DateTime<Self::Timezone>;
    fn end_of_day(&self) -> DateTime<Self::Timezone>;
    /// get end of week. the default week start day is Monday.
    fn end_of_week(&self) -> DateTime<Self::Timezone>;
    /// get end of week given specific week start day.
    fn end_of_week_with_start_day(&self, week_start_day: &WeekStartDay)
        -> DateTime<Self::Timezone>;
    fn end_of_month(&self) -> DateTime<Self::Timezone>;
    fn end_of_quarter(&self) -> DateTime<Self::Timezone>;
    fn end_of_year(&self) -> DateTime<Self::Timezone>;
}

pub trait DateTimeNow {
    type Timezone: TimeZone;
    fn beginning_of_minute(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_hour(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_day(&self) -> DateTime<Self::Timezone>;
    /// get beginning of week. the default week start day is Monday.
    fn beginning_of_week(&self) -> DateTime<Self::Timezone>;
    /// get beginning of week given specific week start day.
    fn beginning_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone>;

    fn beginning_of_month(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_quarter(&self) -> DateTime<Self::Timezone>;
    fn beginning_of_year(&self) -> DateTime<Self::Timezone>;

    fn end_of_minute(&self) -> DateTime<Self::Timezone>;
    fn end_of_hour(&self) -> DateTime<Self::Timezone>;
    fn end_of_day(&self) -> DateTime<Self::Timezone>;
    fn end_of_week(&self) -> DateTime<Self::Timezone>;
    fn end_of_week_with_start_day(&self, week_start_day: &WeekStartDay)
        -> DateTime<Self::Timezone>;
    fn end_of_month(&self) -> DateTime<Self::Timezone>;
    fn end_of_quarter(&self) -> DateTime<Self::Timezone>;
    fn end_of_year(&self) -> DateTime<Self::Timezone>;

    fn week_of_year(&self) -> u32;
}

impl<T> TimeZoneNow for T
where
    T: TimeZone,
{
    type Timezone = T;

    fn now(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self)
    }

    fn beginning_of_minute(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_minute()
    }

    fn beginning_of_hour(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_hour()
    }

    fn beginning_of_day(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_day()
    }

    fn beginning_of_week(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_week()
    }

    fn beginning_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone> {
        Utc::now()
            .with_timezone(self)
            .beginning_of_week_with_start_day(week_start_day)
    }

    fn beginning_of_month(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_month()
    }

    fn beginning_of_quarter(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_quarter()
    }

    fn beginning_of_year(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).beginning_of_year()
    }

    fn end_of_minute(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_minute()
    }

    fn end_of_hour(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_hour()
    }

    fn end_of_day(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_day()
    }

    fn end_of_week(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_week()
    }

    fn end_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone> {
        Utc::now()
            .with_timezone(self)
            .end_of_week_with_start_day(week_start_day)
    }

    fn end_of_month(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_month()
    }

    fn end_of_quarter(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_quarter()
    }

    fn end_of_year(&self) -> DateTime<Self::Timezone> {
        Utc::now().with_timezone(self).end_of_year()
    }
}

impl<T: TimeZone> DateTimeNow for DateTime<T> {
    type Timezone = T;

    fn beginning_of_minute(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms(local_date_time.hour(), local_date_time.minute(), 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_hour(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms(local_date_time.hour(), 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_day(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms(0, 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_week(&self) -> DateTime<Self::Timezone> {
        self.beginning_of_week_with_start_day(&WeekStartDay::Monday)
    }

    fn beginning_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone> {
        let prec_day = match week_start_day {
            WeekStartDay::Monday => self.weekday().number_from_monday() - 1,
            WeekStartDay::Sunday => self.weekday().num_days_from_sunday(),
        };
        let time: DateTime<T> = self.clone().sub(Duration::days(prec_day as i64));
        let succ_local_date_time = time.naive_local();
        let time5 = NaiveDate::from_ymd(
            succ_local_date_time.year(),
            succ_local_date_time.month(),
            succ_local_date_time.day(),
        )
        .and_hms(0, 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_month(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(local_date_time.year(), local_date_time.month(), 1)
            .and_hms(0, 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_quarter(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let month = match local_date_time.month() {
            1..=3 => 1u32,
            4..=6 => 4u32,
            7..=9 => 7u32,
            _ => 10u32,
        };
        let time5 = NaiveDate::from_ymd(local_date_time.year(), month, 1).and_hms(0, 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn beginning_of_year(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(local_date_time.year(), 1, 1).and_hms(0, 0, 0);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn end_of_minute(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms_nano(
            local_date_time.hour(),
            local_date_time.minute(),
            59,
            999999999,
        );
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn end_of_hour(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms_nano(local_date_time.hour(), 59, 59, 999999999);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn end_of_day(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 = NaiveDate::from_ymd(
            local_date_time.year(),
            local_date_time.month(),
            local_date_time.day(),
        )
        .and_hms_nano(23, 59, 59, 999999999);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn end_of_week(&self) -> DateTime<Self::Timezone> {
        self.end_of_week_with_start_day(&WeekStartDay::Monday)
    }

    fn end_of_week_with_start_day(
        &self,
        week_start_day: &WeekStartDay,
    ) -> DateTime<Self::Timezone> {
        let succ_day = match week_start_day {
            WeekStartDay::Monday => 7 - self.weekday().number_from_monday(),
            WeekStartDay::Sunday => 7 - self.weekday().number_from_sunday(),
        };
        let time: DateTime<T> = self.clone().add(Duration::days(succ_day as i64));
        let succ_local_date_time = time.naive_local();
        let time5 = NaiveDate::from_ymd(
            succ_local_date_time.year(),
            succ_local_date_time.month(),
            succ_local_date_time.day(),
        )
        .and_hms_nano(23, 59, 59, 999999999);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn end_of_month(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let (year, month) = if local_date_time.month() == Month::December.number_from_month() {
            (
                local_date_time.year() + 1,
                Month::January.number_from_month(),
            )
        } else {
            (local_date_time.year(), local_date_time.month() + 1)
        };

        let time5 = NaiveDate::from_ymd(year, month, 1).and_hms(0, 0, 0);
        self.timezone()
            .from_local_datetime(&time5)
            .unwrap()
            .sub(Duration::nanoseconds(1))
    }

    fn end_of_quarter(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let (year, month) = match local_date_time.month() {
            1..=3 => (local_date_time.year(), 4u32),
            4..=6 => (local_date_time.year(), 7u32),
            7..=9 => (local_date_time.year(), 10u32),
            _ => (local_date_time.year() + 1, 1u32),
        };
        let time5 = NaiveDate::from_ymd(year, month, 1).and_hms(0, 0, 0);
        self.timezone()
            .from_local_datetime(&time5)
            .unwrap()
            .sub(Duration::nanoseconds(1))
    }

    fn end_of_year(&self) -> DateTime<Self::Timezone> {
        let local_date_time = self.naive_local();
        let time5 =
            NaiveDate::from_ymd(local_date_time.year(), 12, 31).and_hms_nano(23, 59, 59, 999999999);
        self.timezone().from_local_datetime(&time5).unwrap()
    }

    fn week_of_year(&self) -> u32 {
        self.iso_week().week()
    }
}

#[cfg(test)]
mod test {
    use crate::{DateTimeNow, WeekStartDay};
    use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Timelike, Utc, Weekday};

    #[test]
    fn test_end_of_day() {
        use chrono::FixedOffset;

        use crate::TimeZoneNow;
        let offset = FixedOffset::east(60 * 60 * 8);
        let x = offset.end_of_day();
        assert_eq!(23, x.hour());
        assert_eq!(59, x.minute());
        assert_eq!(59, x.second());
    }

    #[test]
    fn test_beginning_of_week() {
        let naive_date_time = NaiveDate::from_ymd(2021, 7, 21).and_hms(0, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();

        assert_eq!(Weekday::Mon, date_time.beginning_of_week().weekday());
        assert_eq!(19, date_time.beginning_of_week().day());

        assert_eq!(
            Weekday::Sun,
            date_time
                .beginning_of_week_with_start_day(&WeekStartDay::Sunday)
                .weekday()
        );
        assert_eq!(
            18,
            date_time
                .beginning_of_week_with_start_day(&WeekStartDay::Sunday)
                .day()
        );
    }

    #[test]
    fn test_end_of_week() {
        let naive_date_time = NaiveDate::from_ymd(2021, 7, 21).and_hms(0, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();

        assert_eq!(Weekday::Sun, date_time.end_of_week().weekday());
        assert_eq!(25, date_time.end_of_week().day());

        assert_eq!(
            Weekday::Sat,
            date_time
                .end_of_week_with_start_day(&WeekStartDay::Sunday)
                .weekday()
        );
        assert_eq!(
            24,
            date_time
                .end_of_week_with_start_day(&WeekStartDay::Sunday)
                .day()
        );
    }

    #[test]
    fn test_week_of_year() {
        let naive_date_time = NaiveDate::from_ymd(2012, 1, 1).and_hms(1, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();
        assert_eq!(52, date_time.week_of_year());

        let naive_date_time = NaiveDate::from_ymd(2014, 12, 29).and_hms(1, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();
        assert_eq!(1, date_time.week_of_year());
        let naive_date_time = NaiveDate::from_ymd(2021, 7, 21).and_hms(1, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();
        assert_eq!(29, date_time.week_of_year());
    }

    #[test]
    fn test_leap_year() {
        let naive_date_time = NaiveDate::from_ymd(2024, 2, 10).and_hms(0, 0, 1);
        let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();
        let time = date_time.end_of_month();

        assert_eq!(29, time.day());
    }
}
