from .converter import get_scenarios, preprocess_rounD_scenarios, preprocess_highD_scenarios, convert_rounD_scenario, convert_highD_scenario, preprocess_inD_scenarios, convert_inD_scenario, preprocess_terasim_scenarios, convert_terasim_scenario, convert_sinD_scenario, preprocess_sinD_scenarios, get_scenarios_sinD

__all__ = {
    'rounD': (get_scenarios, preprocess_rounD_scenarios, convert_rounD_scenario),
    'sinD': (get_scenarios_sinD, preprocess_sinD_scenarios, convert_sinD_scenario),
    'highD': (get_scenarios, preprocess_highD_scenarios, convert_highD_scenario),
    'inD': (get_scenarios, preprocess_inD_scenarios, convert_inD_scenario),
    'terasim': (get_scenarios, preprocess_terasim_scenarios, convert_terasim_scenario)
}

def build_converter(dataset):
    get_scenarios, convert_scenario_func, preprocess =  __all__[dataset]
    # Creating a lambda function that presets the 'dataset' argument for convert_scenario
    # convert_with_preset = lambda scenario, version: convert_scenario_func(scenario, version, dataset)

    return get_scenarios, convert_scenario_func, preprocess
