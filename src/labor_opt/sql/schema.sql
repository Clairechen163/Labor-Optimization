-- Labor management warehouse (SQLite dialect; standard SQL apart from date functions).
-- Core HR entities, scheduled and actual time, pay, and the rules used for compliance checks.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS job_role (
    role_id             TEXT PRIMARY KEY,
    role_name           TEXT NOT NULL,
    base_wage           REAL NOT NULL,           -- per hour
    supervisor_role_id  TEXT REFERENCES job_role(role_id)
);

CREATE TABLE IF NOT EXISTS employee (
    employee_id    TEXT PRIMARY KEY,
    first_name     TEXT NOT NULL,
    last_name      TEXT NOT NULL,
    role_id        TEXT NOT NULL REFERENCES job_role(role_id),
    contract_type  TEXT NOT NULL CHECK (contract_type IN ('FT', 'PT')),
    department     TEXT NOT NULL,
    hire_date      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shift_schedule (
    schedule_id   INTEGER PRIMARY KEY,
    employee_id   TEXT NOT NULL REFERENCES employee(employee_id),
    role_id       TEXT NOT NULL REFERENCES job_role(role_id),    -- role worked (may differ if cross-trained)
    work_date     TEXT NOT NULL,
    template      TEXT NOT NULL,
    shift_start   TEXT NOT NULL,                                 -- ISO timestamp
    shift_end     TEXT NOT NULL,
    premium       REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS attendance (
    attendance_id  INTEGER PRIMARY KEY,
    schedule_id    INTEGER NOT NULL REFERENCES shift_schedule(schedule_id),
    employee_id    TEXT NOT NULL REFERENCES employee(employee_id),
    work_date      TEXT NOT NULL,
    clock_in       TEXT,
    clock_out      TEXT,
    status         TEXT NOT NULL CHECK (status IN ('present', 'absent', 'imputed_clock_out')),
    worked_hours   REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS payroll (
    payroll_id      INTEGER PRIMARY KEY,
    employee_id     TEXT NOT NULL REFERENCES employee(employee_id),
    period_start    TEXT NOT NULL,
    period_end      TEXT NOT NULL,
    regular_hours   REAL NOT NULL,
    overtime_hours  REAL NOT NULL,
    base_pay        REAL NOT NULL,
    premium_pay     REAL NOT NULL,
    overtime_pay    REAL NOT NULL,
    gross_pay       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS performance (
    performance_id     INTEGER PRIMARY KEY,
    employee_id        TEXT NOT NULL REFERENCES employee(employee_id),
    evaluation_date    TEXT NOT NULL,
    performance_score  REAL NOT NULL CHECK (performance_score BETWEEN 1 AND 5),
    comments           TEXT
);

CREATE TABLE IF NOT EXISTS training (
    training_id    INTEGER PRIMARY KEY,
    employee_id    TEXT NOT NULL REFERENCES employee(employee_id),
    training_date  TEXT NOT NULL,
    training_type  TEXT NOT NULL,
    certification  TEXT
);

CREATE TABLE IF NOT EXISTS leave_request (
    leave_id         INTEGER PRIMARY KEY,
    employee_id      TEXT NOT NULL REFERENCES employee(employee_id),
    leave_start      TEXT NOT NULL,
    leave_end        TEXT NOT NULL,
    leave_type       TEXT NOT NULL,
    approval_status  TEXT NOT NULL CHECK (approval_status IN ('approved', 'pending', 'rejected'))
);

CREATE TABLE IF NOT EXISTS labor_rule (
    rule_id             TEXT PRIMARY KEY,
    regulation_type     TEXT NOT NULL,
    max_hours_per_week  REAL NOT NULL,
    overtime_threshold  REAL NOT NULL,
    overtime_rate       REAL NOT NULL,
    min_rest_hours      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_schedule_emp_date ON shift_schedule(employee_id, work_date);
CREATE INDEX IF NOT EXISTS ix_attendance_emp_date ON attendance(employee_id, work_date);
CREATE INDEX IF NOT EXISTS ix_payroll_emp ON payroll(employee_id, period_start);

-- ------------------------------------------------------------------ reporting views

-- Scheduled vs actual hours, per shift.
CREATE VIEW IF NOT EXISTS v_shift_variance AS
SELECT s.schedule_id, s.employee_id, s.role_id, s.work_date, s.template,
       (julianday(s.shift_end) - julianday(s.shift_start)) * 24                AS scheduled_hours,
       a.worked_hours,
       a.worked_hours - (julianday(s.shift_end) - julianday(s.shift_start)) * 24 AS variance_hours,
       CASE WHEN a.clock_in IS NOT NULL
            THEN MAX(0, (julianday(a.clock_in) - julianday(s.shift_start)) * 1440) END AS late_minutes,
       a.status
FROM shift_schedule s
JOIN attendance a ON a.schedule_id = s.schedule_id;

-- Weekly hours per employee with a compliance flag against the labor rule.
CREATE VIEW IF NOT EXISTS v_weekly_hours AS
SELECT e.employee_id, e.role_id, e.contract_type,
       date(a.work_date, 'weekday 0', '-6 days')    AS week_start,
       ROUND(SUM(a.worked_hours), 2)                  AS worked_hours,
       ROUND(MAX(0, SUM(a.worked_hours) - r.overtime_threshold), 2) AS overtime_hours,
       -- 30-minute grace so clock-out drift of a few minutes is not reported as a breach
       CASE WHEN SUM(a.worked_hours) > r.max_hours_per_week + 0.5 THEN 1 ELSE 0 END AS over_max_hours
FROM attendance a
JOIN employee e ON e.employee_id = a.employee_id
CROSS JOIN (SELECT * FROM labor_rule LIMIT 1) r
GROUP BY e.employee_id, week_start;

-- Consecutive worked shifts with less rest than the rule allows.
CREATE VIEW IF NOT EXISTS v_rest_violations AS
WITH ordered AS (
    SELECT employee_id, work_date, clock_in, clock_out,
           LAG(clock_out) OVER (PARTITION BY employee_id ORDER BY clock_in) AS prev_clock_out
    FROM attendance WHERE clock_in IS NOT NULL
)
SELECT o.employee_id, o.work_date, o.prev_clock_out, o.clock_in,
       ROUND((julianday(o.clock_in) - julianday(o.prev_clock_out)) * 24, 2) AS rest_hours
FROM ordered o
CROSS JOIN (SELECT * FROM labor_rule LIMIT 1) r
WHERE o.prev_clock_out IS NOT NULL
  AND (julianday(o.clock_in) - julianday(o.prev_clock_out)) * 24 < r.min_rest_hours;

-- Daily labor cost and hours by role worked.
CREATE VIEW IF NOT EXISTS v_labor_cost_daily AS
SELECT s.work_date, s.role_id,
       COUNT(*)                                                      AS shifts,
       ROUND(SUM(a.worked_hours), 1)                                 AS worked_hours,
       ROUND(SUM(a.worked_hours * e_role.base_wage * s.premium), 0)  AS labor_cost
FROM shift_schedule s
JOIN attendance a ON a.schedule_id = s.schedule_id
JOIN employee e ON e.employee_id = s.employee_id
JOIN job_role e_role ON e_role.role_id = e.role_id
GROUP BY s.work_date, s.role_id;

-- Absence rate by role.
CREATE VIEW IF NOT EXISTS v_absenteeism AS
SELECT s.role_id,
       COUNT(*)                                                  AS scheduled_shifts,
       SUM(CASE WHEN a.status = 'absent' THEN 1 ELSE 0 END)      AS absences,
       ROUND(AVG(CASE WHEN a.status = 'absent' THEN 1.0 ELSE 0 END), 4) AS absence_rate
FROM shift_schedule s JOIN attendance a ON a.schedule_id = s.schedule_id
GROUP BY s.role_id;
