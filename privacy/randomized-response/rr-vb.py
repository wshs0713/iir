#!/usr/bin/env python
# -*- coding: utf-8 -
"""Randomized Response (Collapsed Variational Bayesian)

This code is available under the MIT License.
(c)2021 Nakatani Shuyo / Cybozu Labs Inc.

Usage:
    experiments
    $ python rr-vb.py 100

    summary
    $ python rr-vb.py
"""

import sys, os, itertools, json
from multiprocessing import Pool
import numpy
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=3, suppress=True)

def iterproducts(list1, list2, cycles):
    for _ in range(cycles):
        iters = itertools.product(list1, list2)
        for x in iters: yield x

true_prob = numpy.array([0.1, 0.2, 0.3, 0.4])
legend = [str(x) for x in true_prob]
D = true_prob.size
#true_prob /= true_prob.sum()

p = 1/5
pii = (1 + (D - 1) * p) / D # P(Y=i|X=i)
pij = (1 - p) / D # P(Y=j|X=i)
P = pij * numpy.ones((D, D)) + (pii - pij) * numpy.eye(D)

datapath = "rr-vb.txt"

def variational_bayes(args):
    print("start:(%d, %.2f)" % args)
    N, alpha = args
    true_count = numpy.array(N * true_prob, dtype=int)
    true_count[-1] += N - true_count.sum()
    true_cum = numpy.cumsum(true_count)

    predicts = []
    for _ in range(10):
        Y = numpy.concatenate([numpy.random.choice(D, n, p=pb) for pb, n in zip(P, true_count)])

        X = numpy.random.random((N,D)) # P(X_n)
        X = (X.T/X.sum(axis=1)).T
        c = X.sum(axis=0)

        pre = c / c.sum()
        index = numpy.arange(N)
        for epoch in range(200):
            #numpy.random.shuffle(index)
            for n in index:
                c -= X[n,:]
                x = P[:,Y[n]] * (alpha + c)
                z = X[n,:] = x / x.sum()
                c += z
            pi = c / c.sum()
            if ((pi - pre)**2).sum() < 1e-7: break
            pre = pi
        #print(epoch, pi)
        predicts.append((c/N).tolist())
    print("end:(%d, %.2f)" % args)
    return {"N":N, "alpha":alpha, "predicts":predicts}

if __name__ == '__main__':
    if len(sys.argv)>1:
        I = int(sys.argv[1])
        tasks = iterproducts([10000, 1000, 100], [1.0, 0.1, 0.01], I)
        with Pool(os.cpu_count()-1) as pool:
            for outputs in pool.imap(variational_bayes, tasks):
                #print(outputs)
                with open(datapath, "a") as f:
                    json.dump(outputs, f)
                    f.write("\n")
    else:
        data = dict()
        with open(datapath) as f:
            for s in f:
                x = json.loads(s)
                N = x["N"]
                alpha = x["alpha"]
                predicts = x["predicts"]
                key = (N, alpha)
                if key in data:
                    data[key].extend(predicts)
                else:
                    data[key] = predicts

        cm = plt.get_cmap("tab10")
        for key, predicts in data.items():
            N, alpha = key
            print("VB: N=%d, alpha=%.2f, 1.true, 2.mean, 3.std, 4-5.95%%, 6.median (trials=%d)" % (N, alpha, len(predicts)))
            predicts = numpy.array(predicts)
            print(numpy.vstack(([true_prob, numpy.mean(predicts, axis=0), numpy.std(predicts, axis=0)], numpy.quantile(predicts, [0.025,0.975,0.5], axis=0))))

            start = numpy.min(predicts)
            end = numpy.max(predicts)
            xseq = numpy.arange(start, end, 0.001)
            pdfs = [gaussian_kde(predicts[:,i])(xseq) for i in range(D)]
            bins = 50
            step = (end - start)/bins

            plt.hist(predicts, bins=numpy.arange(start, end, step), density=True)
            plt.legend(legend)
            for i in range(D):
                plt.plot(xseq, pdfs[i], color=cm.colors[i], linewidth=0.5)
            plt.title("VB : N = %d, alpha = %.2f" % (N, alpha))
            plt.tight_layout()
            plt.savefig("rr-vb-%d-%.2f.png" % (N, alpha))
            plt.close()

