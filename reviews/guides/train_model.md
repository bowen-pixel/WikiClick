# train_model.py
## Goal: Set up the interface to handle the train datas, so that data's can be used by Pytroch

### Packages :
- Pyorch
- JSON: the raw datas fetched from WiKi API endpoints

### Functions:
| Function | Attributes | Output | Purpose | Extra |
| :------ | :------ | :----- | :------- | :------- |
| `load_pairs` | `path: DataSet`; `graph_id` | `(selected, counts)` | Processes all the raw input data; remove unreachable ones; remove duplicates| Have to use BFS priorly to get the optimal solutions between pairs|
| `make_tensors` | `pairs`, `vectors` | `Pytorch tensors(X, Y)` | Convert the already loaded pairs and optimal distance into two tensors which can be computed by Pytorch; X is matrix whose rows are consisted of numerical attributes between the pairs; Y is the correspond optimal distance| Require we already compute the optimal distance between arbitrary pairs from the DataSet|
| `mean_absolute_error` | `pairs`, `vectors` | Mean Absolute difference between Y and predicted optimal $\hat{Y}$ | MAE measures the difference between the redicted and actual path length; In later attained by $\hat{Y}$ = model(x)| $\text{NiL}$ |
| `train` | $\text{NiL}$ | $\text{NiL}$ | $\text{NiL}$| We decompose into following helper functiosn |
|  `prepare_data`| `data`,`dimensions` | `graph_id`, `goals`, `vectors` | graph_id identifies which graph we are working on to find optimal path; goals capture the path-connected vertices; vectors stores numerical attributes for each vertex in Graph | Different from load_pair, since each vertex is assigned to a tensor already  |
|  `load_split`| `split`,`graph_id`,`data`,`goals`,`vector` | Pytorch tensors (X,Y); counts | SPlit the datas into training data, validation, testing |  |
|  `create_model`| `dimensions`, `hidden` |  | Initialize the neural model; dimensions refer to the input layer and hidden is the middle layer that the input datas have to bypass to return as output |  |
|  `train_one_epoch`| `model`, `optimizer`, `loss_fn` |  `model`, `optimizer`, `loss_fn`| model is the DistanceModel we used to train the datas; optimizer is the coefficients (the gradients) we used to update every epoch; loss_fn measures how predicted ones differ from the BFS solution |  |
|  `fit_model`| `model`, `loader`, `optimizer`, `loss_fn`,`x_validation`, `y_validation`, `epochs` | `best_state`, `best_epoch`, `best_error`,` history` | Evaluate the validation MAE and Select the best fit model according to multiple epochs|  |
|  `evaluate_and_save`|  |  `metrics`| We use the best fit model from above and train on the training data and save the evaluation statistics |  |
|  `train`| `seed` |  | seed is used to assign random weight initially to the pairs |  |

