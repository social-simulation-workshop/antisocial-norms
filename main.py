import argparse
import csv
import itertools
import multiprocessing
import numpy as np
import os

from args import ArgsConfig



class Agent:
    _ids = itertools.count(0)

    def __init__(self) -> None:
        self.id = next(self._ids)

        self.theta = np.random.randint(0, 10)
        self.w = 0
        self.v = 0


class AntiSocialNorms:

    def __init__(self, args: argparse.ArgumentParser,
                 alpha: float, mu: float, lambda_rival: float, verbose=True) -> None:
        Agent._ids = itertools.count(0)
        self.verbose = verbose

        self.args = args

        # exp1
        if not self.args.exp2:
            self.alpha = alpha
            self.mu = mu
            self.lambda_rival = lambda_rival
        
        # exp2
        else:
            self.alpha = np.random.uniform()
            self.mu = mu
            self.lambda_rival = lambda_rival

        self.ags = [Agent() for _ in range(args.N)]
        self.n = sum([ag.w for ag in self.ags])
        self.V = sum([ag.v for ag in self.ags])
    

    def get_iw(self, n:int):
        ret = self.args.g - self.args.c + \
              self.mu * (1 - self.lambda_rival*n/(n+1))
        return ret
    

    def get_utility(self, ag:Agent, n:int):
        ret = self.args.g*n + \
              ag.w * (self.args.g - self.args.c + self.mu * (1-self.lambda_rival*n/(n+1)))
        return ret


    def simulate(self) -> tuple:
        for _ in range(self.args.n_rounds):
            np.random.shuffle(self.ags)
            changes = False
            for ag in self.ags:
                ori_n = self.n
                ori_V = self.V

                # compute ag's inclination to work (IW)
                n = self.n - ag.w
                iw = self.get_iw(n)
                utility_n = self.get_utility(ag, n)

                # choose to work or shirk
                cond = self.alpha*self.V + (1-self.alpha)*iw
                self.n -= ag.w
                if cond > 0:
                    ag.w = 1 # work
                else:
                    ag.w = 0 # shirk
                self.n += ag.w
                
                # choose to promote, abstain , oppose
                n = self.n - ag.w
                n_up = min(n + ag.theta, self.args.N-1)
                n_low = max(n - ag.theta, 0)
                utility_n_up = self.get_utility(ag, n_up)
                utility_n_low = self.get_utility(ag, n_low)
                payoff_pro = utility_n_up - utility_n - self.args.e
                payoff_opp = utility_n_low - utility_n - self.args.e

                self.V -= ag.v
                if max(payoff_pro, payoff_opp) <= 0:
                    ag.v = 0 # abstain
                else:
                    if payoff_pro > payoff_opp:
                        ag.v = 1 # promote
                    elif payoff_pro < payoff_opp:
                        ag.v = -1 # oppose
                    else:
                        ag.v = 0 # abstain
                self.V += ag.v

                if ori_n != self.n or ori_V != self.V:
                    changes = True
            
            if not changes:
                break

        # results
        part = self.n / self.args.N
        pro = sum([1 for ag in self.ags if ag.v == 1]) / self.args.N
        opp = sum([1 for ag in self.ags if ag.v == -1]) / self.args.N
        abst = sum([1 for ag in self.ags if ag.v == 0]) / self.args.N
        
        return {'part': part,
                'pro': pro,
                'opp': opp,
                'abst': abst}


def replicate(args: argparse.ArgumentParser,
              alpha: float, mu: float, lambda_rival: float) -> dict:
    part, pro, opp, abst = list(), list(), list(), list()
    for _ in range(args.n_replications):
        game = AntiSocialNorms(args, alpha=alpha, mu=mu, lambda_rival=lambda_rival)
        res = game.simulate()
        part.append(res['part'])
        pro.append(res['pro'])
        opp.append(res['opp'])
        abst.append(res['abst'])
    
    if game.verbose:
        if not args.exp2:
            print("lambda {:.1f} alpha {:.2f} mu {:2d} || part: {:.3f}; pro: {:.3f}; opp: {:.3f} abst: {:.3f}".format( \
                lambda_rival, alpha, int(mu),
                sum(part)/len(part), sum(pro)/len(pro), sum(opp)/len(opp), sum(abst)/len(abst)))
        else:
            print("lambda {:.1f} alpha N/A mu {:2d} || part: {:.3f}; pro: {:.3f}; opp: {:.3f} abst: {:.3f}".format( \
                lambda_rival, int(mu),
                sum(part)/len(part), sum(pro)/len(pro), sum(opp)/len(opp), sum(abst)/len(abst)))
    
    return {'part': sum(part)/len(part),
            'pro': sum(pro)/len(pro),
            'opp': sum(opp)/len(opp)}


def catch_data_multiprocessing(args, alpha, mu, lambda_rival, log_data):
    res = replicate(args, alpha, mu, lambda_rival)
    log_data.append([alpha, mu, lambda_rival,
                        res["part"], res["pro"], res["opp"]])


if __name__ == "__main__":
    parser = ArgsConfig()
    args = parser.get_args()

    # csv
    if not args.exp2:
        out_path = os.path.join(os.getcwd(), "lambda_{:.1f}_rndSeed_{}_nRepli_{}.csv".format( \
                                args.lambda_rival, args.rnd_seed, args.n_replications))
    else:
        out_path = os.path.join(os.getcwd(), "exp2_rndSeed_{}_nRepli_{}.csv".format( \
                                args.rnd_seed, args.n_replications))
    out_file = open(out_path, "w", newline="")
    out_writer = csv.writer(out_file)
    out_writer.writerow(["alpha", "mu", "lambda", "participation", "promote", "oppose"])

    # multiprocessing
    if args.multiprocessing:
        n_cpus = multiprocessing.cpu_count()
        print("cpu count: {}".format(n_cpus))

        manager = multiprocessing.Manager()
        log_data = manager.list()

        args_list = list()
        if not args.exp2:
            for alpha in np.arange(args.alpha_low_bound, args.alpha_up_bound, args.alpha_interval):
                for mu in np.arange(args.mu_low_bound, args.mu_up_bound, args.mu_interval):
                    args_list.append([args, alpha, mu, args.lambda_rival, log_data])
        else:
            for lambda_rival in np.arange(args.lambda_low_bound, args.lambda_up_bound, args.lambda_interval):
                for mu in np.arange(args.mu_low_bound, args.mu_up_bound, args.mu_interval):
                    args_list.append([args, None, mu, lambda_rival, log_data])
        pool = multiprocessing.Pool(n_cpus+2)
        pool.starmap(catch_data_multiprocessing, args_list)

        for data in log_data:
            out_writer.writerow(data)

    # single process
    else:
        if not args.exp2:
            for alpha in np.arange(args.alpha_low_bound, args.alpha_up_bound, args.alpha_interval):
                for mu in np.arange(args.mu_low_bound, args.mu_up_bound, args.mu_interval):
                    np.random.seed(args.rnd_seed)
                    res = replicate(args, alpha, mu, args.lambda_rival)
                    out_writer.writerow([alpha, mu, args.lambda_rival, res["part"], res["pro"], res["opp"]])
        else:
            for lambda_rival in np.arange(args.lambda_low_bound, args.lambda_up_bound, args.lambda_interval):
                for mu in np.arange(args.mu_low_bound, args.mu_up_bound, args.mu_interval):
                    res = replicate(args, None, mu, lambda_rival)
                    out_writer.writerow([None, mu, lambda_rival, res["part"], res["pro"], res["opp"]])
