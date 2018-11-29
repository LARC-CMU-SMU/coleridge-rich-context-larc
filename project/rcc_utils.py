import json
import os

from rcc_conf import (
    PARSED_PUB_PATH, DATASET_CITATION_FILE, CACHE_PUB_FILE,
    RESEARCH_METHODS_FILE
)


def json_from_file(filename):
    data = None
    with open(filename, 'r') as f:
        data = json.load(f)
    if data is None:
        raise ValueError('Error loading json from {}.'.format(filename))
    return data


def _exact_research_method_match(pub, rmethods_graph):
    """ This function returns all matching research methods labels in all
        sections of a parsed publication.
        Input:
            - pub: parsed publication
            - rmethod_graph: research method graph
        Output:
            return a dictionary representing how many times
            a particular concept appear in the publication
    """
    if pub['metadata']['sections'] is None:
        return None
    result = []
    sections = pub['metadata']['sections']
    for concept in rmethods_graph:
        if concept['@type'] != 'skos:Concept':
            continue

        # count concept prefLabel
        # count all concept altLabel, if available
        pref_label = concept['skos:prefLabel']['@value'].lower()
        if 'skos:altLabel' not in concept:
            alt_labels = []
        else:
            if type(concept['skos:altLabel']) is list:
                alt_labels = [alt_label['@value'].lower()
                              for alt_label in concept['skos:altLabel']]
            elif type(concept['skos:altLabel']) is dict:
                alt_labels = [concept['skos:altLabel']['@value'].lower()]
            else:
                raise ValueError('unknown type: {}'
                                 .format(type(concept['skos:altLabel'])))
        count = 0
        for sec in sections:
            text = sec['text'].lower()
            count += text.count(pref_label)
            for alt_label in alt_labels:
                count += text.count(alt_label)

        if count > 0:
            result.append({'id': concept['@id'], 'count': count})
    return result if result else None


def load_rcc_train_dataset(data_path, force_compute=False):
    """ This function loads rcc parsed publications for training.dataset.
        We assume that research methods file and `data_set_citations.json`
        are available in the folder.
        We also expect the parsed publications (parsed using
        AllenAI science parser) are available too.
        The loaded dataset is cached into a json file, and subsequent data
        load utilizes cache file.
        Input:
            - data_path: data folder containing `publications.json`,
                         `files/json/` folder containing parsed publication.
            - force_compute: if True, then we recompute everything and ignore
                             cache file.
        Output:
            return a list of parsed publications
    """
    cache_train_file = data_path + CACHE_PUB_FILE
    if os.path.isfile(cache_train_file) and not force_compute:
        return json_from_file(cache_train_file)

    parsed_pub_path = data_path + PARSED_PUB_PATH
    files = [os.path.join(parsed_pub_path, f)
             for f in os.listdir(parsed_pub_path)]
    parsed_pubs = [json_from_file(f) for f in files]
    citations = json_from_file(data_path + DATASET_CITATION_FILE)
    citing_pubs = set([citation['publication_id'] for citation in citations])
    rmethods_graph = json_from_file(RESEARCH_METHODS_FILE)['@graph']

    count = 0
    for pub in parsed_pubs:
        # Information whether the publication is citing a dataset or not
        if int(pub['name'][:-4]) in citing_pubs:
            pub['citing'] = 1
        else:
            pub['citing'] = 0

        # Research method information
        pub['rmethods'] = _exact_research_method_match(pub, rmethods_graph)

        count += 1
        if count % 200 == 0:
            print('{} publication processed.'.format(count))

    # cache consolidated dataset
    with open(cache_train_file, 'w') as f:
        f.write(json.dumps(parsed_pubs))
    return parsed_pubs


def load_rcc_test_dataset(data_path, force_compute=False):
    """ This function loads rcc parsed publications for test.dataset.
        We assume that research methods file is available in the folder.
        We also expect the parsed publications (parsed using
        AllenAI science parser) are available too.
        The loaded dataset is cached into a json file, and subsequent data
        load utilizes cache file.
        Input:
            - data_path: data folder containing `publications.json`,
                         `files/json/` folder containing parsed publication.
            - force_compute: if True, then we recompute everything and ignore
                             cache file.
        Output:
            return a list of parsed publications
    """
    cache_train_file = data_path + CACHE_PUB_FILE
    if os.path.isfile(cache_train_file) and not force_compute:
        return json_from_file(cache_train_file)

    parsed_pub_path = data_path + PARSED_PUB_PATH
    files = [os.path.join(parsed_pub_path, f)
             for f in os.listdir(parsed_pub_path)]
    parsed_pubs = [json_from_file(f) for f in files]
    rmethods_graph = json_from_file(RESEARCH_METHODS_FILE)['@graph']

    count = 0
    for pub in parsed_pubs:
        # Research method information
        pub['rmethods'] = _exact_research_method_match(pub, rmethods_graph)

        count += 1
        if count % 200 == 0:
            print('{} publication processed.'.format(count))

    # cache consolidated dataset
    with open(cache_train_file, 'w') as f:
        f.write(json.dumps(parsed_pubs))
    return parsed_pubs


def load_rcc_cache_dataset(data_path):
    """ This function loads cache dataset which contain parsed publication
        information and additional contextual information such as citation
        information and research methods
    """
    cache_file = data_path + CACHE_PUB_FILE
    if os.path.isfile(cache_file):
        return json_from_file(cache_file)
    else:
        raise ValueError('Cache file {} does not exist.'.format(cache_file))
