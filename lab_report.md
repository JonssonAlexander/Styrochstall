Main task: Evaluate maximum possible free time for Styr&Ställ

Ok so initially a bit wierd to download the data. 
1. I had to get the API from Mobility data (talk about how mobility data is a provider for this kind of info for alot of different ride sharing platforms (such as dott))

You do this by filtering for country code SE on https://github.com/MobilityData/gbfs/blob/master/systems.csv

There are docs but they´re tedious: https://github.com/nextbike/api-doc/blob/master/README.md

When you find what you need you then need to run the script populate_data.py. So the thing here is to have the correct api web socket. Took me a while to get the correct one but eventually found the right one.  

To fetch data: python populate_data.py --iterations 5 --interval-seconds 120

Calibrate using this:
python3.11 -m styrstell.cli calibrate --config configs/calibration.yaml --no-fetch

And simulate using this:
python3.11 -m styrstell.cli simulate --config configs/simulation.yaml

run this for OD matrix:
python3.11 plot_od_map.py --top-k 175 --min-flow 1 \
  --annotate --label-top 10 \
  --boundary data/external/gothenburg_boundary.geojson


After a bit of digging, we managed to find that looking at each bike, we could see which station it has been to (?, think so or vice versa). anyways, now it's just a matter of back-tracking and we should be able to find the correct bike rides (indirectly including travel times as well if we take a high enough frequency)

run this:
python3.11 plot_od_map.py \
  --flow-path data/models/observed_od_matrix.parquet \
  --probability-path data/models/observed_od_probability.parquet \
  --boundary data/external/gothenburg_boundary.geojson \
  --annotate --label-top 10 \
  --output-path reports/observed_od_map.html


this for the bike numbers is better:

and we run it using this 
python3.11 -m styrstell.cli infer-maps-trips --config configs/calibration.yaml

python3.11 -m styrstell.cli observed-od --config configs/calibration.yaml --trips data/processed/maps_trips.parque


Now with this we can 1. watch the difference between MAPS and GBFS 2. interactive OD map 3. we can see a heatmap of departures/arrivals 4. Still working on rebalancing routes 

# Calibrate as before (GBFS models remain untouched)
python3.11 -m styrstell.cli calibrate --config configs/calibration.yaml

# Infer trips from MAPS (bike_list with bike_numbers fallback)
python3.11 -m styrstell.cli infer-maps-trips \
  --config configs/calibration.yaml \
  --min-duration-seconds 0 --min-distance-km 0.0

# Build observed OD products from those MAPS trips
python3.11 -m styrstell.cli observed-od \
  --config configs/calibration.yaml \
  --trips data/processed/maps_trips.parquet

# Interactive OD map (station names, Folium)
python3.11 plot_od_map.py \
  --flow-path data/models/observed_od_matrix.parquet \
  --probability-path data/models/observed_od_probability.parquet \
  --boundary data/external/gothenburg_boundary.geojson \
  --annotate --label-top 10 \
  --output-path reports/observed_od_map.html

# Station-activity heatmap (departures/arrivals/net; e.g. departures shown)

This is a nice picture of activity (yellow dots that become larger if there are more departures/arrivals)

python3.11 plot_station_activity_map.py \
  --trips data/processed/maps_trips.parquet \
  --metric departures \
  --output reports/station_activity_departures.html

# GBFS vs MAPS comparison (HTML + GIF, station names)
python3.11 compare_gbfs_maps.py \
  --config configs/calibration.yaml \
  --output-parquet data/processed/gbfs_maps_station_counts.parquet \
  --output-html reports/gbfs_vs_maps.html \
  --output-gif reports/gbfs_vs_maps.gif

# Detect and map suspected rebalancing routes
python3.11 detect_rebalancing.py \    
  --config configs/calibration.yaml \
  --threshold 5 \
  --events-output data/processed/rebalancing_events.parquet \
  --routes-output data/processed/rebalancing_routes.parquet \
  --output-map reports/rebalancing_routes.html

NOTE You can run it but it'll neeed some extra love for it to properly work. 

to fix the rebalancing, this was what i prompted:
note that rebalancing does not have to have a specific route. It just have to show where either a large departure spike or arrival spike happended. Also include the timestamp. Then use like dijkstras or something to come up with the shortest route from that and call that our car route.

so basically we made a dijkstra on a k nearest neighbour station graph and skips routes if they are faster than 35km/h. 

python3.11 detect_rebalancing.py \
  --config configs/calibration.yaml \
  --threshold 5 \
  --max-speed-kmph 35 \
  --min-speed-kmph 5 \
  --depot-lat 57.708 \
  --depot-lon 11.974 \
  --events-output data/processed/rebalancing_events.parquet \
  --routes-output data/processed/rebalancing_routes.parquet \
  --output-map reports/rebalancing_routes.html


Alright, now we have a github action running every 5 mins (or supposed to run every 5 mins). So if everything goes right, we should see results in a few days.

If you just do it like the following code, it automatically creates what you need to be able to create the maps:

python3.11 -m styrstell.cli infer-maps-trips --config configs/calibration.yaml
python3.11 -m styrstell.cli observed-od --config configs/calibration.yaml --trips data/processed/maps_trips.parquet 

Then run the plotting scripts.


From the data we have, what to do?
1. Want to estimate if the system survives change in trip length policy but too long between data gather intervals? Let's see if we can get it running on a local/good server. Nevertheless, if we have the Markov state space model, we should be able to bootstrap where people will go, and then use google maps data for travel time estimation, add some noise, use the normal amount of departures from each station and simulate the system. Then, we gradually add a percentage of "shoppers", i.e. people who use the service and parks it (maybe able to get that info but prob not necessary), and look at where the threshold for the system is. 


Tried to do something with strongly connected components. this is what i have, i like the last one best but needs further refinments
Recompute analytics (choose your own fraction):

python3.11 graph_diagnostics.py --tidy-od data/models/observed_od_tidy.parquet --edge-threshold 0.2 --absorbing-output data/processed/absorbing_states.parquet --cycles-output data/processed/strongly_connected_components.parquet --dependency-output data/processed/station_dependencies.parquet --flow-summary-output data/processed/station_flow_summary.parquet

Refresh the existing SCC map:

python3.11 graph_components_map.py --station-info data/processed/station_information.parquet --absorbing data/processed/absorbing_states.parquet --components data/processed/strongly_connected_components.parquet --output reports/graph_components_map.html

I think this to see graph dependencies:
python3.11 graph_dependency_map.py --station-info data/processed/station_information.parquet --dependency-data data/processed/station_dependencies.parquet --flow-summary data/processed/station_flow_summary.parquet --share-threshold 0.2 --cross-share-threshold 0.3 --output reports/station_dependency_map.html


TODO  Isolate time of day activity per week day 
TODO  Hook up a weather api for each station and see if amount of trips are correlated with rain.



For the markov simulation:

python3 build_markov_chain.py \
  --maps-trips data/processed/maps_trips.parquet \
  --stations data/processed/station_information.parquet \
  --output data/processed/markov_transitions.parquet \
  --time-granularity 1H --smoothing 0.05


python3 prepare_simulation_data.py \
  --trips data/processed/maps_trips.parquet \
  --station-counts data/processed/gbfs_maps_station_counts.parquet \
  --rebalance-events data/processed/rebalancing_events.parquet \
  --station-info data/processed/station_information.parquet \
  --snapshot-source MAPS \
  --inventory-strategy mean

python3 update_travel_time_cache.py \
  --transitions data/processed/markov_transitions.parquet \
  --station-info data/processed/station_information.parquet \
  --cache data/processed/travel_time_cache.parquet \
  --probability-threshold 0.01 \
  --mean-speed-kmph 17

python3 run_monte_carlo_simulation.py \
  --transitions data/processed/markov_transitions.parquet \
  --departures data/processed/simulation_departures.parquet \
  --inventory data/processed/simulation_initial_inventory.parquet \
  --rebalance data/processed/simulation_rebalancing_events.parquet \
  --station-info data/processed/station_information.parquet \
  --travel-cache data/processed/travel_time_cache.parquet \
  --replications 200 \
  --mean-speed-kmph 17 \
  --outcomes-output data/processed/monte_carlo_outcomes.parquet \
  --probability-output data/processed/monte_carlo_stockout_probabilities.parquet


python3 plot_stockout_probability_map.py --probabilities data/processed/monte_carlo_stockout_probabilities.parquet --station-info data/processed/station_information.parquet --scenario demand_only --output reports/stockout_probability_map.html --time-granularity 1H   

with refill included:
python3 plot_stockout_probability_map.py --probabilities data/processed/monte_carlo_stockout_probabilities.parquet --station-info data/processed/station_information.parquet --scenario demand_with_refill --output reports/stockout_probability_map_refill.html --time-granularity 1H  
