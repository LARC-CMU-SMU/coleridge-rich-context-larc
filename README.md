# rcc-02
Repository for RCC submission

## Dependencies
Scala for AllenAI Science Parse
Python dependencies:
- python 3.6 via Anaconda
- scikit-learn 0.20.1
- pandas

## Project Structure
This project contains few important folders:
- models: contains all learning models.
- resources: contains resource files such as SAGE research methods and research fields.
- tools: contains external packages such as AllenAI Science Parse.
- data: contains dataset (for training only).

All main codes are under `project/` folder.

## AllenAI Science Parse tool
AllenAI Science Parse parses PDF publication papers into following fields: title, authors, abstract, sections, bibliography.

How to run
```
java -Xmx6g -jar bin/science-parse-cli-assembly-2.0.2-SNAPSHOT.jar -o ../data/input/files/json/ ../data/input/files/pdf/

java -Xmx6g -jar tools/science-parse-cli-assembly-2.0.2-SNAPSHOT.jar  -m models/scienceparse/productionModel-v9.dat -b models/scienceparse/productionBibModel-v7.dat -g models/scienceparse/gazetteer-v5.json -o ../data/input/files/json2/ ../data/input/files/pdf/
```

## CLI Snippets

Research fields recommendation
```
python rfields_rec.py --input_dir ../data/input/ --output_dir ../data/output/
```

### Following snippets are examples for training models

Train dataset detection
```
python dataset_detect_train.py --input_dir data/train_test/  --output models/dataset_detect.model
```
Train research fields recommendation models
```
python rfields_rec_train.py --input data/rfield_10.json --output models/rfields_rec.model
```

