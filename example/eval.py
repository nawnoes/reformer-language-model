
def evaluate(model, dev_data_path):
  # predict = os.path.join(gdrive_path, "finetuning/data/korquad/KorQuAD_v1.0_train.json")
  predict = dev_data_path
  eval_examples = read_squad_examples(input_file=predict,
                                      is_training=False,
                                      version_2_with_negative=False)
  eval_features = convert_examples_to_features(examples=eval_examples,
                                               tokenizer=tokenizer,
                                               max_seq_length=max_seq_length,
                                               doc_stride=doc_stride,
                                               max_query_length=max_query_length,
                                               is_training=False)

  all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
  sampler = SequentialSampler(dataset)
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=predict_batch_size)

  logger.info("***** Evaluating *****")
  logger.info("  Num features = %d", len(dataset))
  logger.info("  Batch size = %d", predict_batch_size)

  model.eval()
  model.to(device)
  all_results = []
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)

  logger.info("Start evaluating!")
  for input_ids, input_mask, segment_ids, example_indices in tqdm(dataloader, desc="Evaluating", leave=True,
                                                                  position=0):
    input_ids = input_ids.to(device)
    with torch.no_grad():
      batch_start_logits, batch_end_logits = model(input_ids)
    for i, example_index in enumerate(example_indices):
      start_logits = batch_start_logits[i].detach().cpu().tolist()
      end_logits = batch_end_logits[i].detach().cpu().tolist()
      eval_feature = eval_features[example_index.item()]
      unique_id = int(eval_feature.unique_id)
      all_results.append(RawResult(unique_id=unique_id,
                                   start_logits=start_logits,
                                   end_logits=end_logits))
  output_prediction_file = os.path.join(output_dir, "predictions.json")
  output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
  write_predictions(eval_examples, eval_features, all_results,
                    n_best_size, max_answer_length,
                    False, output_prediction_file, output_nbest_file,
                    None, False, False, 0.0)

  with open(predict) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']

  with open(os.path.join(output_dir, "predictions.json")) as prediction_file:
    predictions = json.load(prediction_file)
  logger.info(json.dumps(evaluate(dataset, predictions)))
