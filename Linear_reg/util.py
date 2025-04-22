import time
import json
import os 
import argparse
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import numpy as np 

def gen_data():
# y = mx +c 
# add in a bit of noise (in consistent)
    x = np.linspace(0,10,100000)
    w = 2.65
    b= 0.8
    noise = np.random.randn(100000)

    y = x*w + b+ noise
    return x,y

# original data is created now time for linear regression on python 

def lr(x,y):
    w=0.0
    b=0.0
    l_r=0.01
    epochs=1000
    N = len(x)
    for _ in range(epochs):
        y_pred = x*w +b
        error = y_pred - y 

        w -= l_r *(np.dot(error,x)/N)
        b -= l_r *(error.sum()/N)

    y_pred=w*x+b
    return w,b

def main():
    parser = argparse.ArgumentParser(description="Util FIle")
    
    parser.add_argument("--gen", action="store_true", help="Generates data")
    parser.add_argument("--plot", action="store_true", help="Plots the results")
    parser.add_argument("--lr", action="store_true", help="Linear regression on Original Data")
    args= parser.parse_args()

    if args.gen:
        x,y=gen_data()
        x.tofile('x.bin')
        y.tofile('y.bin')

    if args.lr:
        y= np.fromfile("y.bin",dtype=np.float64)
        x= np.fromfile("x.bin",dtype=np.float64)
        start = time.time()
        w,b = lr(x,y)
        end = time.time()
        results ={}
        if os.path.exists("results.json"):
            with open("results.json","r")as f:
                results = json.load(f)

        results["python"]={
                    "w":round(w,4),"b":round(b,4),"time":round(end-start,4)
                }
        with open("results.json","w")as f:
            json.dump(results,f,indent=4)
    
    if args.plot:
        x = np.fromfile("x.bin", dtype=np.float64)
        y = np.fromfile("y.bin", dtype=np.float64)

        with open("results.json", "r") as f:
            results = json.load(f)

        # 1️⃣ Runtime bar chart
        langs = list(results.keys())
        times = [results[lang]["time"] for lang in langs]

        plt.figure(figsize=(6, 4))
        plt.bar(langs, times, color="orange")
        plt.ylabel("Time (s)")
        plt.title("Execution Time per Implementation")
        plt.grid(axis='y')
        plt.show(block=False)

        # Regression lines with subset of data
        # subset_indices = np.random.choice(len(x), size=200, replace=False)
        # x_subset = x[subset_indices]
        # y_subset = y[subset_indices]
        #
        # plt.figure(figsize=(8, 6))
        # plt.scatter(x_subset, y_subset, label="Subset Data", color="blue", s=10)
        #
        # for lang, values in results.items():
        #     w = values["w"]
        #     b = values["b"]
        #     t = values["time"]
        #
        #     y_pred = w * x_subset + b
        #     plt.plot(x_subset, y_pred, label=f"{lang.title()} (t={t:.4f}s)")
        #
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # plt.title("Linear Regression Comparison")
        # plt.grid(True)
        plt.show(block=False)
        input("Press Enter to close plots...")
if __name__ == "__main__":
    main()
