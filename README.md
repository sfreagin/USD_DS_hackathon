# University of San Diego 2023 Data Science Hackathon
#### Stephen Reagin
This is a repository for a submission to the inaugural (2023) University of San Diego Data Science Hackathon, organized by Ready Tensor: https://app.readytensor.ai/

The task is multiclass classification, specifically predicting factors that may have affected automobile drivers upon a vehicle accident. The **Fatality Analysis Reporting System** (FARS) dataset comes from the National Highway Traffic Safety Administration (NHTSA), although here it is [sourced from another GitHub repository](https://github.com/readytensor/rt-datasets-fars/tree/main) maintained by Abhyuday Desai, founder and CEO of Ready Tensor.

## Original Code Source and Structure of Repository
The structure of this repository is designed to work with containers, i.e. to "Dockerize" a model so that a submission can be run on the Ready Tensor platform. The original structure and code comes from this repository maintained by Ready Tensor: https://github.com/readytensor/usd-hackathon-fars-template-python-scripts/tree/main

The main difference between the original code from Ready Tensor and our submission code is:
* We have included two Jupyter Notebooks:
  * One for EDA purposes, helping to identify feature variables which have a significant impact on the target distribution.
  * One for building simple models with a reduced feature dataset
* The `train.py` files and `predict.py` files have been altered:
  * The feature dataset now undergoes a different preprocessing path, including:
      * Reducing the number of features
      * Converting "deaths" to a binary variable
  * Using a `RandomForestClassifier` from the `sklearn` library
 
Otherwise, the bulk of the `train.py` and `predict.py` files are largely the same as had previously been created by Ready Tensor. We expect to eventually generate our own custom models following an original Dockerization of code files, but this borrowing of large code portions (from [this repo](https://github.com/readytensor/usd-hackathon-fars-template-python-scripts/tree/main) also linked above) was necessary for us to submit contributions in the allotted 48-hour window.

## Task
The task is to predict one of three potential factors that could have affected the driving behavior leading to a fatal crash. The categories and baseline expectations are:
* `other` 57%
* `drunk_driver_involved` 26%
* `speeding_driver_involved` 17%

We instantiated a Random Forest Classifier with default parameters from the sklearn library and included the following features (descriptions come from Ready Tensor repo):

* `a_ped_f`:  Whether pedestrian fatality involved in crash
* `a_roll`: Whether vehicle rollover involved in crash
* `day_week`: Day of the week when crash occurred
* `a_dow_type`: Day of week type: weekday (M-F) or weekend (Sat-Sun)
* `a_tod_type`: Time of day time: daytime (6 am to 6 pm), nighttime (6 pm to 6 am)
* `a_region`: Region (made up of states) where crash occurred
* `a_ru`: Rural or urban
* `a_intsec`: Whether crash occurred at an intersection or not
* `a_roadfc`: Type of road (interstate, local, etc.)
* `a_junc`: Identifies if crash occurred in or proximity to junction or interchange area of two or more roadways
* `a_relrd`: Identifies area of roadway where crash occurred (on, off, shoulder, median, etc.)
* `age`: Age of driver
* `pernotmvit`: Number of persons not in motor vehicles in-transport
* `a_ped`: Whether crash involved a pedestrian
* `a_body`: Vehicle body type (automobile, light trucks, mediu/high trucks, buses, etc.)
* `owner`: Type of registered owner of vehicle in crash
* `deaths`: Number of fatalities in vehicle
* `deformed`: Extent of damage to vehicle

## Results

Our best submitted model has:
* Accuracy score of 64%
  * https://en.wikipedia.org/wiki/Accuracy_and_precision
* Weighted Average F1 score of 62%
  * https://en.wikipedia.org/wiki/F-score
* Weighted Average AUC-ROC of 0.75
  * https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  
## Future Developments

Future developments may require better feature engineering, including techniques like dimensionality reduction and correlation analysis. We also want to hypertune parameters across multiple learning algorithms for classification to identify generalizable models that are reliably accurate.

Please feel free to borrow any original code in this repository, and also be sure to cross-check the Ready Tensor repository to ensure you have permission to use their code.

---

MIT License

Copyright (c) [2023] [Stephen F. Reagin]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
