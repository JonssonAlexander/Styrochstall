# Data Quality Summary

## Station Information

Stations: 135

Capacity stats (min/median/mean/max): 0 / 0 / 0.0 / 0

|   station_id | name                    |   short_name |     lat |     lon |   region_id | is_virtual_station   | rental_uris.android                          | rental_uris.ios                              | rental_uris.web            |   capacity |
|-------------:|:------------------------|-------------:|--------:|--------:|------------:|:---------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------|-----------:|
|     28031717 | Dalgångsgatan           |        41100 | 57.6734 | 11.9793 |         658 | False                | https://app.nextbike.net/station?id=28031717 | https://app.nextbike.net/station?id=28031717 | https://nxtb.it/p/28031717 |          0 |
|     28031764 | Krokslätts fabriker (A) |        41102 | 57.6712 | 12.0096 |         658 | False                | https://app.nextbike.net/station?id=28031764 | https://app.nextbike.net/station?id=28031764 | https://nxtb.it/p/28031764 |          0 |
|     28031786 | Lackarebäck (A)         |        41103 | 57.6662 | 12.0119 |         658 | False                | https://app.nextbike.net/station?id=28031786 | https://app.nextbike.net/station?id=28031786 | https://nxtb.it/p/28031786 |          0 |
|     28031883 | Södra Porten            |        41104 | 57.6672 | 12.0172 |         658 | False                | https://app.nextbike.net/station?id=28031883 | https://app.nextbike.net/station?id=28031883 | https://nxtb.it/p/28031883 |          0 |
|     28031924 | Mölndals innerstad (A)  |        41106 | 57.6561 | 12.0168 |         658 | False                | https://app.nextbike.net/station?id=28031924 | https://app.nextbike.net/station?id=28031924 | https://nxtb.it/p/28031924 |          0 |

## Station Panel

Rows: 3915 | Stations: 135 | Window: 2025-10-01 13:05:00+00:00 → 2025-10-01 15:25:00+00:00

|      |   num_bikes_available |   num_docks_available |
|:-----|----------------------:|----------------------:|
| min  |                0      |                     0 |
| max  |               52      |                     0 |
| mean |               10.3254 |                     0 |


## Station Flows

Rows: 3915 | Negative counts: 0 | Rebalancing flags: 8

|      |   departures |   arrivals |
|:-----|-------------:|-----------:|
| mean |     0.163814 |   0.147382 |
| max  |     5.33333  |   3.33333  |


## Demand Intensity (λ)

|      |   lambda_departures |   lambda_arrivals |
|:-----|--------------------:|------------------:|
| mean |           0.0302163 |         0.0275181 |
| max  |           0.633333  |         0.5       |


Top departure intensities:

station_id
28045600     0.633333
28045513     0.533333
30921680     0.433333
28045845     0.400000
28045704     0.333333
31100774     0.300000
556609101    0.233333
31054478     0.233333
28045897     0.233333
551862781    0.200000


## Travel-Time Distribution

Time bins: 6 | bins off by >5%: 0

First bins:

| time_bin   |   duration_bin_start |   duration_bin_end |   probability |   count |
|:-----------|---------------------:|-------------------:|--------------:|--------:|
| 00:00:00   |                    0 |                  5 |    0.00847458 |       0 |
| 00:00:00   |                    5 |                 10 |    0.00847458 |       0 |
| 00:00:00   |                   10 |                 15 |    0.855932   |       1 |
| 00:00:00   |                   15 |                 20 |    0.00847458 |       0 |
| 00:00:00   |                   20 |                 25 |    0.00847458 |       0 |
| 00:00:00   |                   25 |                 30 |    0.00847458 |       0 |
| 00:00:00   |                   30 |                 35 |    0.00847458 |       0 |
| 00:00:00   |                   35 |                 40 |    0.00847458 |       0 |
| 00:00:00   |                   40 |                 45 |    0.00847458 |       0 |
| 00:00:00   |                   45 |                 50 |    0.00847458 |       0 |


## OD Matrices

Slices: 405 | off-by->5%: 147

Examples of mismatched probability sums:

slot_start                 origin   
2025-10-01 13:00:00+00:00  168224366    0.0
                           195150203    0.0
                           28031717     0.0
                           28031764     0.0
                           28031786     0.0

