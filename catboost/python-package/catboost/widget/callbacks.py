try:
    from xgboost.callback import TrainingCallback as XGBTrainingCallback
except:
    class XGBTrainingCallback:
        pass


from IPython.display import display

from .metrics_plotter import MetricsPlotter

class XGBPlottingCallback(XGBTrainingCallback):
    '''XGBoost callback with metrics plotting widget from CatBoost
    '''
    
    def __init__(self, total_iterations: int):
        self.plotter = None        
        self.total_iterations = total_iterations

    def after_iteration(self, model, epoch, evals_log):
        
        data_names = evals_log.keys()
        # if more than one sample is passed, consider first as train sample 
        first_train = (
            all(['valid' in data_name.lower() for data_name in data_names]) 
            and len(data_names) > 1
        )
        
        for data_name, metrics_info in evals_log.items():
            if "train" in data_name.lower() or first_train:
                train = True
                first_train = False
            elif "valid" in data_name.lower() or "test" in data_name.lower():
                train = False
            else:
                raise Exception("Unexpected sample name during evaluation")
                
            metrics = {name: values[-1] for name, values in metrics_info.items()}
               
            if self.plotter is None:
                names = list(metrics.keys())
                self.plotter = MetricsPlotter(names, names, self.total_iterations)
                display(self.plotter._widget)
            
            self.plotter.log(epoch, train, metrics)
        
        # False to indicate training should not stop.
        return False


def lgbm_plotting_callback():
    """LightGBM callback with metrics plotting widget from CatBoost
    """
    
    plotter = None
    
    def _init(env):
        train_metrics = []
        test_metrics = []
        
        for item in env.evaluation_result_list:
            assert len(item) == 4, "Plotting was run in not suppored mode"
            data_name, eval_name = item[:2]
            
            if "train" in data_name.lower():
                train_metrics.append(eval_name)
            elif "valid" in data_name.lower() or "test" in data_name.lower():
                test_metrics.append(eval_name)
            else:
                raise Exception("Unexpected sample name during evaluation")
             
        nonlocal plotter
        plotter = MetricsPlotter(
            train_metrics, test_metrics, env.end_iteration - env.begin_iteration)
        
        display(plotter._widget)
    
    def _callback(env):
        if plotter is None:
            _init(env)
            
        metrics = {"train": {}, "test": {}}
        
        for item in env.evaluation_result_list:
            data_name, eval_name, result = item[:3]
            
            if "train" in data_name.lower():
                metrics["train"][eval_name] = result
            elif "valid" in data_name.lower() or "test" in data_name.lower():
                metrics["test"][eval_name] = result
            else:
                raise Exception("Unexpected sample name during evaluation")
                
        plotter.log(
            env.iteration - env.begin_iteration, 
            train=True, 
            metrics=metrics["train"]
        )
        plotter.log(
            env.iteration - env.begin_iteration, 
            train=False, 
            metrics=metrics["test"]
        )
            
    _callback.order = 20
    return _callback