import glob, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


pd.options.mode.chained_assignment = None  # default='warn'

tt = datetime.now()
date_to_print = tt.strftime("%Y%m%d")

list_files = glob.glob("./annotated/*.txt")

good = []
watch = []
artefact = []
for txt_file in list_files:
    with open(txt_file) as f:
        ID_tmp = Path(txt_file).stem.split(".diff")[0]
        if "good_candidate" in txt_file:
            good.append(ID_tmp)
        elif "watch_list" in txt_file:
            watch.append(ID_tmp)
        elif "artefact" in txt_file:
            artefact.append(ID_tmp)


df_good = pd.DataFrame({"SNID": np.array(good)})
df_good.to_csv(f"./annotated/good_candidates_{date_to_print}.csv")

df_watch = pd.DataFrame({"SNID": np.array(watch)})
df_watch.to_csv(f"./annotated/watch_candidates_{date_to_print}.csv")

artefact = pd.DataFrame({"SNID": np.array(artefact)})
artefact.to_csv(f"./annotated/artefacts_candidates_{date_to_print}.csv")

