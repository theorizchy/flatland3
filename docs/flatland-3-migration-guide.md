# Moving Flatland 2 code to Flatland 3 - Common changes

## 1. Line Generators

`schedule_generators` are now renamed to `line_generators`

This is because Flatland 2 had no concept of a timetable for allowed departures of expected arrival times. `line_generators` now generate the data that `schedule_generators` used to (start-end station pairs for agents).

If your code used `sparse_schedule_generator`, replace it with `sparse_line_generator`, the behaviour slightly varies, but it should be similar for the most high level code.

---

## 2. Change from `max_rails_in_city` to `max_rail_pairs_in_city`

`sparse_rail_generator` has an input parameter `max_rails_in_city` replaced with `max_rail_pairs_in_city`. You can replace the original input number that was being used by dividing it by 2. This is to ensure every city has a minimum of 2 rails which we keep as a constraint for balancing inflow and outflow of the trains.

---

## 3. Waiting State

Flatland 3 introduces a `WAITING` state for every train, in which all actions will be ignored if the elapsed timesteps is less than the train's earliest departure. 

---

## 4. Change from `RailAgentStatus` to `TrainState`

The Flatland `env.step` function has been refactored to include a state machine for better intepretability of the functions for people who like to read, modify, and extend the functionality of the code as per their requirements.

The older `RailAgentStatus` and the attribute `agent.moving` is now changed to `TrainState`.

This can now be accessed as `agent.state` instead of `agent.status`.

For more information on state machine please check https://flatland.aicrowd.com/environment/state_machine.html

---

## 5. Removal of `agent.malfunction_data` and `agent.speed_data`

Flatland 2 used to represent malfunctions in a dictionary `agent.malfunction_data`, this is now moved to the class `agent.malfucntion_handler` which keeps the **same functionality** but refactored for better readability. 

The common changes needed for older code using malfunction data are:

`agent.malunction_data['malfunction']` -> `agent.malfunction_handler.malfunction_downd_counter`

`agent.malfunction_data['nr_malfunctions']` -> `agent.malfunction_handler.num_malfunctions`

Train speed and partial fractional position in cells was stored in the dictionary `agent.speed_data`, this is now moved to the class `agent.speed_counter`, which keeps the **same functionality** but refactored as a counter instead of fractional position.

The common changes needed for older code using speed_data are:

`agent.speed_data['speed']` -> `agent.speed_counter.speed`
`agent.speed_data['position_fraction']` -> `agent.speed_counter.counter`, where position fraction indicated the fraction of completed movement, the speed counter counts upto `agent.speed_counter.max_count` before resetting.

---

## 6. Removal of `random` and `complex` generator classes

The genrator classes `complex_rail_generator`, `complex_schedule_genrator`, `random_rail_generator`, `random_schedule_generator` are now deprecated and removed from the environment. Please use `sparse_rail_generator` and `sparse_line_generator` instead.

--- 

## 7. Safe minimum env size is 30x30
While its possible to generate smaller environments for exploration and understanding, we recommend using 30x30 as the minimum size to satisfy new constrains on the rail generation.


