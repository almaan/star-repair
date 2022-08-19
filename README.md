# repair


## Setup and installation

First, clone this repository to your own machine. Open the terminal, enter a directory that you want the cloned repository to reside within and do:

```sh
git clone git@github.com:almaan/star-repair.git
```

Next, and this requires that you have
[conda](https://docs.conda.io/en/latest/miniconda.html) installed, create a new
conda environment based on the `yaml` file included in the repository. Also make sure to activate this environment *now* and _whenever_ you want to run `repair`.


```sh
conda env create --file conda/repair.yml
conda activate repair
```

<hr>
:warning: if you're experiencing issues with the package `lap` (for example, the former complaining about an array size mismatch). Then you've somehow, most likely, compiled `lap` with the wrong `numpy` version. One solution that should work for this problem is to execute the following commands in your terminal (whilst being in the conda environment):

```sh
pip uninstall lap
pip uninstall numpy
pip install numpy==1.21.4
pip install lap==0.4.0  --force-reinstall --ignore-installed --no-cache-dir
```
<hr>

Next, install `repair`:

```sh
python ./setup.py install
```


You should now be able to run `repair` as a CLI application. Note, you may have
to source your rc file again (or just open a new terminal window). Test this by:

```sh
repair analyze -h
```

This should display a message like this:

```sh
usage: repair analyze [-h] -i INPUT [-o OUTDIR] [-ap A_PATTERN]
                      [-bp B_PATTERN] [-lr LEARNING_RATE] [-e EPOCHS]
                      [-l1 L1_REGULARIZATION] [-mxo MIN_X_OBS] [-x X]
                      [-mc MIN_COUNTS] [-b BASE] [-nb N_BINS] [-sa]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTDIR, --outdir OUTDIR
  -ap A_PATTERN, --a_pattern A_PATTERN
  -bp B_PATTERN, --b_pattern B_PATTERN
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -e EPOCHS, --epochs EPOCHS
  -l1 L1_REGULARIZATION, --l1_regularization L1_REGULARIZATION
  -mxo MIN_X_OBS, --min_x_obs MIN_X_OBS
  -x X, --x X
  -mc MIN_COUNTS, --min_counts MIN_COUNTS
  -b BASE, --base BASE
  -nb N_BINS, --n_bins N_BINS
  -sa, --save_adata
```

Congrats, you've successfully installed `repair`!


## Analyze

Assuming that you've installed `repair` we can now analyze our data. To analyze
an input (`INPUT_FILE`) file we'll use the following structure of the command
(do not run): 

```sh
repair analyze -i INPUT_FILE -o OUTDIR -ap A_PATTERN -bp B_PATTERN -mxo 10 -x 1
-b 100 -nb 1
```

There are some variables (all uppercase) here which will change depending on
your file. We'll be using a file called `test-counts.tsv` located at
`/tmp/demo/indata`; making the full path to this file
`/tmp/demo/indata/test-counts.tsv`, this will replace `INPUT_FILE`. 

Looking at the first five rows and columns of this file we have:
```shape
                 IGLCclone0   IGHGclone1  IGLCclone2  IGLCclone3  IGKCclone5
GACGGGATGTCTTATG-1_1  0           0           0           0           0
CACATAAGGCGACCGT-1_1  0           0           0           0           0
TGACCCAACTCACATT-1_1  0           0           0           0           0
ACTTATCTGATCTATA-1_1  0           0           0           0           0

```

The two different kind of chains that we have are IGL* and IGH*, where *
represents some arbitrary text. This will be the basis of our "chain patterns" (`A_pattern`
and `B_pattern`). These patterns are supposed to be provided in _regex_ style,
so we thus have `A_pattern` = `^(IGH)` and `B_pattern` = `^(IGL)`. Finally,
we'll set the output directory (`OUTDIR`) to `/tmp/demo/output`. Needless to
say, you should change these variables to whatever is befitting to your own
data. Our final command thus become: 


```sh
repair analyze -i /tmp/demo/indata/test-counts.tsv -o /tmp/demo/output -ap "^(IGH)" -bp "^(IGL)" -mxo 10 -x 1
-b 100 -nb 1
```

The command `-mxo` states how many observations (e.g., spots) that should have a
minimum of `x` counts of a given chain for that chain to be included in the
analysis; the value of `x` is determined by the argument `-x` (here 1). To
clarify, if we were to use `-mxo 20 -x 2`, for a chain to be included in the
analysis there would have to be 20 observations (e.g., spots) with _at least two
counts of that chain.

Once the analysis has been executed, you will have two files, both starting with
a 20 digit long tag (this tag is different for every analysis):

```sh
20220818170159685663_analysis_result.tsv
20220818170159685663_loss_history.dat
```

in the `TAG_analysis_result.tsv` file you'll find the result of the pairing, it
looks something like:

```sh
	      chainA	chainB	      score
Pair_0	IGHGclone1	IGLCclone0	0.6102368
Pair_1	IGHGclone6	IGLCclone2	0.50854367
Pair_2	IGHAclone8	IGLCclone10	0.54590046
Pair_3	IGHGclone19	IGLCclone58	0.24616973

```


The
`TAG_loss_history.dat` file gives the loss value for each epoch, you can inspect
this with a graphing software if you want to make sure that the model has
converged (if not, change the number of epochs with the `-e` flag). In the next
section, we'll refer to the results file as `RESULTS_FILE`.


## Evaluate

`repair` can also be used to evaluate your result, to do this, simply run:

```sh
repair evaluate -i RESULT_FILE -o OUTDIR -gt GROUND_TRUTH_FILE -ci A_PATTERN_RES
B_PATTERN_RES -ct A_PATTERN_GT B_PATTERN_GT
```

Again, we'll replace some of the variables here with values pertaining to our
specific data:

The `GROUND_TRUTH_FILE` will be a file listing the ground truth pairs, the file
we're using in this demo looks like:

```sh
                chainA	chainB
clonotype321	IGHAclone1171	IGKCclone2749
clonotype2   	IGHAclone136	IGKCclone26
clonotype100	IGHAclone1990	IGKCclone2501
clonotype31	    IGHAclone2066	IGKCclone1096
```

This path to this file is `/tmp/demo/input/ground-truth.tsv`.


The `X_PATTERN_RES` variables (X={A,B}) is the column names of the A and B chain
in the results file (look at the end of the previous section - "Analyze" - and
you'll see this is `chainA` and `chainB`). `X_PATTERN_GT` is the equivalent for
the ground truth, conveniently  the same (`chainA` and `chainB`).

Thus, the final evaluation command becomes:

```sh
repair evaluate -i /tmp/demo/output/20220818170159685663_analysis_result.tsv -o
/tmp/demo/output -gt /tmp/demo/input/ground-truth.tsv -ci chainA chainB -ct
chainA chainB
```

This will generate a few more files, with the same tag as before, in your output folder:

```sh
20220818170159685663_analysis_result.tsv
20220818170159685663_evaluation_result.tsv
20220818170159685663_evaluation_summary_all.tsv
20220818170159685663_evaluation_summary_high.tsv
20220818170159685663_loss_history.dat
```

The `TAG_evauluation_result.tsv` gives you the pairs and whether they are
correct or not, while the `TAG_evaluation_summary_all.tsv` summarize these
results in terms of numbers and accuracy.


