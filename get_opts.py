import sys
import pandas as pd

def main():
    data_file = sys.argv[1]
    job_id = int(sys.argv[2])
    df = pd.read_csv(data_file, sep=",")

    print(" ".join(["{}={}".format(k, v) for k,v in dict(df.iloc[job_id-1]).items()]))

if __name__ == '__main__':
    main()
