import os
import json
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu


# read in file
# get the average response length
# compute statistical significance between files

base_path = "./Base/"
instruct_path = "./Instruct/"

# add open gen file names
open_gen_benchmarks = []
open_gen_benchmarks.append("_en2ga")
open_gen_benchmarks.append("_ga2en")
open_gen_benchmarks.append("_nq")

# example line:
# {"doc_id": 3, "doc": {"en": "Revenue services, refunds and repayments of tax", "ga": "Seirbhísí na gCoimisinéirí Ioncaim, aisíocaíochtaí cánach"}, "target": "Seirbhísí na gCoimisinéirí Ioncaim, aisíocaíochtaí cánach", "arguments": {"gen_args_0": {"arg_0": "Aistrigh Béarla go Gaeilge:\n\nBéarla: Global response event\nGaeilge: Global response event\nBéarla: Each EU/EEA country is responsible for its own national public health policy, including its national immunisation programme and vaccination schedule. Information on the national vaccination schedules in EU/EEA countries can be found in the ECDC Vaccine Scheduler\nGaeilge: Bíonn gach tír AE/LEE freagrach as a mbeartas sláinte phoiblí náisiúnta féin, lena n-áirítear a gclár imdhíonta náisiúnta agus a sceideal vacsaínithe. Is féidir teacht ar na sceidil vacsaínithe náisiúnta sna tíortha AE/LEE i Sceidealóir Vacsaíní ECDC\nBéarla: The European Commission is coordinating a common European response to the coronavirus outbreak.\nGaeilge: Tá freagairt Eorpach chomhchoiteann don ráig COVID-19 á comhordú ag an gCoimisiún Eorpach.\nBéarla: To ensure that banks can continue to lend money, support the economy and help mitigate the economic impacts of the Coronavirus, the European Commission has adopted a banking package.\nGaeilge: Ghlac an Coimisiún Eorpach pacáiste baincéireachta le cinntiú go leanfadh na bainc ar aghaidh ag tabhairt iasachtaí airgid, ag tacú leis an ngeilleagar agus ag cuidiú le tionchair eacnamaíocha an choróinvíris a mhaolú.\nBéarla: Two men in a workshop\nGaeilge: Two men in a workshop\nBéarla: Revenue services, refunds and repayments of tax\nGaeilge:", "arg_1": {"until": ["\n", "\\n", "<s>"], "do_sample": false, "temperature": 0.0}}}, "resps": [[" Seirbhísí ioncaim, aisíocaíochtaí agus aisíocaíochtaí cánachais Béarla: The European Commission has adopted a package of measures to support the economy and help mitigate the economic impacts of the Coronavirus outbreak. The measures include a temporary suspension of VAT on the supply of certain goods and services, a temporary reduction of VAT rates and a temporary reduction of excise duties on certain products. The measures are valid for a period of 3 months, from 1 March 2020 to 31 May 2020. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all"]], "filtered_resps": [" Seirbhísí ioncaim, aisíocaíochtaí agus aisíocaíochtaí cánachais Béarla: The European Commission has adopted a package of measures to support the economy and help mitigate the economic impacts of the Coronavirus outbreak. The measures include a temporary suspension of VAT on the supply of certain goods and services, a temporary reduction of VAT rates and a temporary reduction of excise duties on certain products. The measures are valid for a period of 3 months, from 1 March 2020 to 31 May 2020. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all"], "filter": "none", "metrics": ["bleu", "ter"], "doc_hash": "5c7c101d050cd13edfb1255713f83f146dc60665a2961c2426f2f439cf3b9317", "prompt_hash": "2fff5d2f546a018d78edf7d02033148e61c7255cccfe4952382020f83957f94c", "target_hash": "33d6648e163135a282003c3fb8d4f484f0af48d08f37d532ebdb321d8b0492f9", "bleu": [" Seirbhísí ioncaim, aisíocaíochtaí agus aisíocaíochtaí cánachais Béarla: The European Commission has adopted a package of measures to support the economy and help mitigate the economic impacts of the Coronavirus outbreak. The measures include a temporary suspension of VAT on the supply of certain goods and services, a temporary reduction of VAT rates and a temporary reduction of excise duties on certain products. The measures are valid for a period of 3 months, from 1 March 2020 to 31 May 2020. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all", "Seirbhísí na gCoimisinéirí Ioncaim, aisíocaíochtaí cánach"], "ter": [" Seirbhísí ioncaim, aisíocaíochtaí agus aisíocaíochtaí cánachais Béarla: The European Commission has adopted a package of measures to support the economy and help mitigate the economic impacts of the Coronavirus outbreak. The measures include a temporary suspension of VAT on the supply of certain goods and services, a temporary reduction of VAT rates and a temporary reduction of excise duties on certain products. The measures are valid for a period of 3 months, from 1 March 2020 to 31 May 2020. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all EU countries. The measures are available in all", "Seirbhísí na gCoimisinéirí Ioncaim, aisíocaíochtaí cánach"]}

def get_response_lengths_given_dir(dir):
    # model: benchmrk: [lengths]
    benchmark_response_lengths = {}
    # for filename in each folder
    for filename in os.listdir(dir):
        #print(filename)
        # check if any open_gen_benchmarks match the filename
        if any(b in filename for b in open_gen_benchmarks):
            matched_benchmark = next((b_tag for b_tag in open_gen_benchmarks if b_tag in filename), None)
            # process the file
            #print(dir + filename)
            with open(os.path.join(dir, filename), "r", encoding="utf-8") as f:
                benchmark_response_lengths[matched_benchmark] = []
                #print(f"Processing {filename}")
                # jsonl
                for line in f:
                    # read the as json
                    data = json.loads(line)
                    response = data["resps"][0][0]
                    word_count = len(re.findall(r"\w+", response, flags=re.UNICODE))
                    benchmark_response_lengths[matched_benchmark].append(word_count)
    return benchmark_response_lengths


benchmark_response_lengths = {}
benchmark_response_lengths[base_path] = get_response_lengths_given_dir(base_path)
benchmark_response_lengths[instruct_path] = get_response_lengths_given_dir(instruct_path)
print(benchmark_response_lengths.keys())

# create matrix
df = {"Model": [],
      "Benchmark": [],
      "Mean": [],
      "Lengths": [],
      }
for dir in benchmark_response_lengths:
    for benchmark in benchmark_response_lengths[dir]:
        lengths = benchmark_response_lengths[dir][benchmark]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        df["Model"].append(dir)
        df["Benchmark"].append(benchmark)
        df["Lengths"].append(lengths)
        df["Mean"].append(avg_length)

        # GRAPHS OF DISTRIBUTIONS
        # add model and benchmark to title 
        sns.histplot(lengths, kde=True)
        plt.xlabel("Response Length (words)")
        plt.ylabel("Frequency")
        safe_dir = Path(dir).name or Path(dir).stem or "unknown"
        safe_bench = benchmark.lstrip("_")
        out_dir = Path("Response_Lengths_Distributions") / safe_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_dir}_{safe_bench}.png"
        plt.savefig(out_path)
        plt.close()

df = pd.DataFrame(df)

# calculate man-whitney u
for benchmark in open_gen_benchmarks:
    # filter by df by benchmark
    df_benchmark = df[df["Benchmark"] == benchmark]

    base_lengths = df_benchmark[df_benchmark["Model"] == base_path]["Lengths"].values
    instruct_lengths = df_benchmark[df_benchmark["Model"] == instruct_path]["Lengths"].values


    U1, p = mannwhitneyu(base_lengths[0], instruct_lengths[0], alternative='greater')
    df.loc[df["Benchmark"] == benchmark, "U1"] = U1
    df.loc[df["Benchmark"] == benchmark, "p"] = p



# save df to csv
df = df[["Model", "Benchmark", "Mean", "U1", "p", "Lengths"]]
df.to_csv("benchmark_response_lengths.csv", index=False)





