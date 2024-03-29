# coleridge-rich-context-larc
Code Repository for RCC submission

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
java -Xmx6g -jar tools/science-parse-cli-assembly-2.0.2-SNAPSHOT.jar -o ../data/input/files/json/ ../data/input/files/pdf/
```

## CLI Snippets

Datasets prediction
```
python datasets_predict.py --input_dir ../data/input/ --output_dir ../data/output/
```

Research methods recommendation
```
python rmethods_rec.py --input_dir ../data/input/ --output_dir ../data/output/
```

Research fields recommendation
```
python rfields_rec.py --input_dir ../data/input/ --output_dir ../data/output/
```

### Following snippets are examples for training models

Train dataset detection
```
python dataset_detect_train.py --input_dir data/train_test/  --output models/dataset_detect.model
```

Train research methods recommendation models
```
python rmethods_rec_train.py --input data/rmethod_ctx_train_mx100.json --output models/rmethods_rec.model
```

Train research fields recommendation models
```
python rfields_rec_train.py --input data/rfield_10.json --output models/rfields_rec.model
```

