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
  --output-parquet data/processed/rebalancing_events.parquet \
  --output-map reports/rebalancing_routes.html

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


TODO  Isolate time of day activity per week day 
TODO  Hook up a weather api for each station and see if amount of trips are correlated with rain.