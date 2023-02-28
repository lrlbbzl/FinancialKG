def extract_entities(test_entities, train_entities):
    # test_entities = to_be_trained_entities
    # train_entities = read_json(Path(DATA_DIR, 'entities.json'))

    for ent_type, ents in test_entities.items():
        test_entities[ent_type] = list(set(ents) - set(train_entities[ent_type]))

    for ent_type in train_entities.keys():
        if ent_type not in test_entities:
            test_entities[ent_type] = []
    return test_entities