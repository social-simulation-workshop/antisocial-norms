import argparse

class ArgsConfig(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        parser = argparse.ArgumentParser()

        # individual
        parser.add_argument("--g", type=float, default=1.0,
            help="the benefit created by each member's work.")
        parser.add_argument("--c", type=float, default=5.0,
            help="the cost of working.")
        parser.add_argument("--e", type=float, default=2.0,
            help="the cost of enforcement.")
        parser.add_argument("--N", type=int, default=10,
            help="the # of members.")
        parser.add_argument("--lambda_rival", type=float, default=0.0,
            help="the rivalness of the incentive.")

        # independent variables
        parser.add_argument("--alpha_low_bound", type=float, default=0.0,
            help="the var alpha is in the range [low_bound, up_bound) with interval alpha_interval.")
        parser.add_argument("--alpha_up_bound", type=float, default=1.0,
            help="the var alpha is in the range [low_bound, up_bound) with interval alpha_interval.")
        parser.add_argument("--alpha_interval", type=float, default=0.02,
            help="the var alpha is in the range [low_bound, up_bound) with interval alpha_interval.")
        
        parser.add_argument("--mu_low_bound", type=float, default=0.0,
            help="the var mu is in the range [low_bound, up_bound) with interval mu_interval.")
        parser.add_argument("--mu_up_bound", type=float, default=50.1,
            help="the var mu is in the range [low_bound, up_bound) with interval mu_interval.")
        parser.add_argument("--mu_interval", type=float, default=1.0,
            help="the var mu is in the range [low_bound, up_bound) with interval mu_interval.")

        parser.add_argument("--lambda_low_bound", type=float, default=0.0,
            help="the var lambda is in the range [low_bound, up_bound) with interval lambda_interval.")
        parser.add_argument("--lambda_up_bound", type=float, default=1.01,
            help="the var lambda is in the range [low_bound, up_bound) with interval lambda_interval.")
        parser.add_argument("--lambda_interval", type=float, default=0.02,
            help="the var lambda is in the range [low_bound, up_bound) with interval lambda_interval.")

        # models variables
        parser.add_argument("--n_rounds", type=int, default=10000,
            help="the number of individuals.")
        parser.add_argument("--n_replications", type=int, default=10,
            help="the number of replication for each unique condition.")
        parser.add_argument("--rnd_seed", type=int, default=1025,
            help="random seed.")
        parser.add_argument("--multiprocessing", nargs="?", const=True, default=False,
            help="random seed.")
        parser.add_argument("--exp2", nargs="?", const=True, default=False,
            help="set on to run experiment 2. if not, run experiment 1.")
        
        self.parser = parser
    

    def get_args(self):
        args = self.parser.parse_args()
        return args
