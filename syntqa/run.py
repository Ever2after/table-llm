import logging
import os
import sys
import torch 
import nltk
import datasets
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from utils.config import ModelArguments, DataTrainingArguments
from importlib import import_module
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Disable wandb during experiment
    # if data_args.max_eval_samples or data_args.max_predict_samples:
    if data_args.max_predict_samples:
        training_args.report_to = []

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name == 'squall':
        task = "./task/squall_plus.py"
        raw_datasets = load_dataset(task, 
                                    plus=data_args.squall_plus, 
                                    downsize=data_args.squall_downsize,
                                    split_id=data_args.split_id,
                                    # download_mode='force_redownload',
                                    ignore_verifications=True)
    elif data_args.dataset_name == 'wikisql':
        task = "./task/wikisql_robut.py"
        raw_datasets = load_dataset(task, 
                                    split_id=data_args.split_id,
                                    perturbation_type=data_args.perturbation_type,
                                    # download_mode='force_redownload',
                                    ignore_verifications=True)
    else:
        raise NotImplementedError

    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = 1024
    config.early_stopping = False
    padding = "max_length" if data_args.pad_to_max_length else False

    # Load main tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )

    if training_args.resume_from_checkpoint != None:
        name = training_args.resume_from_checkpoint
    else:
        name = model_args.model_name_or_path

    if data_args.task.lower() == 'tableqa':
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.task.lower() == 'text_to_sql':
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise NotImplementedError

    # Load dataset preprocess function
    preprocess_module = 'seq2seq.'
    if data_args.task.lower()=='selector':
        preprocess_module += 'selector'
    else:
        preprocess_module += data_args.dataset_name
        if data_args.task.lower()=='tableqa':
            preprocess_module += '_tableqa'
    preprocess_function = import_module(preprocess_module).preprocess_function

    fn_kwargs={"tokenizer":tokenizer, 
                "max_source_length": data_args.max_source_length,
                "max_target_length": data_args.max_target_length,
                "ignore_pad_token_for_loss": data_args.ignore_pad_token_for_loss,
                "padding": padding,
                "input_noise": data_args.input_noise}
        
    if training_args.do_train or data_args.predict_split=='train':
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
                )
    else:
        train_dataset = None
    
    if training_args.do_eval or data_args.predict_split=='dev':
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
                )
    else:
        eval_dataset = None

    if training_args.do_predict:
        if data_args.predict_split=='train':
            predict_dataset = train_dataset
        elif data_args.predict_split=='dev':
            predict_dataset = eval_dataset
            # to be commented
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
        else:
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))

            with training_args.main_process_first(desc="test dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc="Running tokenizer on predict dataset",
                    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Load prepare compute metrics function
    metric_module = 'metric.'
    metric_module += data_args.dataset_name
    if data_args.task.lower()=='tableqa':
        metric_module += '_tableqa'
    prepare_compute_metrics = import_module(metric_module).prepare_compute_metrics

    if training_args.do_train:            
        compute_metrics = prepare_compute_metrics(
            tokenizer=tokenizer, 
            eval_dataset=eval_dataset, 
            stage=None, 
            fuzzy=data_args.postproc_fuzzy_string)
    else:
        dataset_name = data_args.dataset_name
        squall_plus_suffix = '_plus' if data_args.squall_plus else ''
        squall_downsize_suffix = f'_d{data_args.squall_downsize}' if data_args.squall_downsize else ''
        split_id = data_args.split_id
        perturbation_suffix = (
            f'_{data_args.perturbation_type}'
            if dataset_name == 'wikisql' and data_args.predict_split == 'dev' and split_id == 0 and data_args.perturbation_type != 'original'
            else ''
        )
        
        note_suffix = f'_{data_args.save_note}' if data_args.save_note else ''
        if data_args.input_noise is not None:
            note_suffix += f'_noise{data_args.input_noise}'
            
        stage = (
            f'{dataset_name}'
            f'{squall_plus_suffix}'
            f'{squall_downsize_suffix}'
            f'_{data_args.task.lower()}'
            f'_{data_args.predict_split}'
            f'{split_id}'
            f'{perturbation_suffix}'
            f'{note_suffix}'
        )
        
        compute_metrics = prepare_compute_metrics(
            tokenizer=tokenizer, 
            eval_dataset=predict_dataset, 
            stage=stage, 
            fuzzy=data_args.postproc_fuzzy_string)

    # Load trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
 
        b = training_args.per_device_eval_batch_size
        max_len = len(predict_dataset)
        log_probs_sum = []
        log_probs_mean = []
        predictions = []

        for i in tqdm(range(0, max_len, b)):
            end = min(i+b, max_len) 

            input_ids = predict_dataset['input_ids'][i:end]
            input_ids = [item + [tokenizer.pad_token_id]*(data_args.max_source_length-len(item)) for item in input_ids]
            input_ids = torch.tensor(input_ids).to(training_args.device)

            attention_mask = predict_dataset['attention_mask'][i:end]
            attention_mask = [item + [0]*(data_args.max_source_length-len(item)) for item in attention_mask]
            attention_mask = torch.tensor(attention_mask).to(training_args.device)

            # Generate the output using beam search
            gen_outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                num_beams=data_args.num_beams,
                max_length=data_args.val_max_target_length,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # Compute scores for beam search
            scores = model.compute_transition_scores(
                sequences=gen_outputs.sequences,
                scores=gen_outputs.scores,
                beam_indices=gen_outputs.beam_indices,
            )

            output_scores=[]
            for sample in scores.tolist():
                last = len(sample)-1
                while last>0 and sample[last]==0:
                    last -= 1
                output_scores.append(sample[:last+1])

            log_probs_sum += [np.sum(x) for x in output_scores] # sum
            log_probs_mean += [np.mean(x) for x in output_scores] # mean
            tmp = gen_outputs.sequences.cpu().tolist()
            predictions += [item + [tokenizer.pad_token_id]*(data_args.val_max_target_length-len(item)) for item in tmp]

        tmp = predict_dataset["labels"]
        label_ids = [item + [tokenizer.pad_token_id]*(data_args.val_max_target_length-len(item)) for item in tmp]
        
        eval_preds = EvalPrediction(predictions=predictions, label_ids=label_ids)
        acc = compute_metrics(eval_preds, {'log_probs_sum': log_probs_sum, 'log_probs_mean': log_probs_mean})
        print("predict: ", acc)


if __name__ == "__main__":
    main()