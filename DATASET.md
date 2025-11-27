## DATASET

### For VG dataset
The following is from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download the [scene graphs](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catalog.py`.[DATASET.md](../SHA-GCL/DATASET.md)

### For GQA dataset
The following is from [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG).
1. Download the GQA images [Full (20.3 Gb)](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). Extract these images to the file `datasets/gqa/images`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. In order to achieve a representative split like VG150, we manually clean up a substantial fraction of annotations that have poor-quality or ambiguous meanings, and then select Top-200 object classes as well as Top-100 predicate classes by their frequency, thus establishing the GQA200 split. You can download the annotation file from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kwwKFbdBB3ZU3c49?e=06qeZc).

### For OIV6 dataset
The following is from [PySGG](https://github.com/SHTUPLUS/PySGG).

The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).
The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 
You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3)
The dataset dir contains the `images` and `annotations` folder. Link the `open_image_v4` and `open_image_v6` dir to the `/datasets/openimages` then you are ready to go.
