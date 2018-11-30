# rcc-02
Repository for RCC submission

## Dependencies
Scala for AllenAI Science Parse
Python dependencies:
- python 3.6 via Anaconda
- scikit-learn
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
```
