import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import os
import config
from model import BertNER
# from model import CRFModel
from metrics import f1_score, bad_case
from transformers import BertTokenizer

from torch.cuda.amp import autocast as ac
from attack_train_utils import FGM, PGD
import class_balanced_loss
from class_balanced_loss import CB_loss

def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    #-----------------fp16
    scaler = None
    use_fp16 = True
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    #-----------------fp16 end
    #-----------------cb_loss
    # use_CB_loss =True
    #-----------------cb_loss end
    #---------------fgm
    fgm, pgd = None, None
    attack_train_mode = 'pgd'
    if attack_train_mode == 'fgm':
        fgm = FGM(model=model)
    elif attack_train_mode == 'pgd':
        pgd = PGD(model=model)

    pgd_k = 5
    #---------------fgm end

    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    #-------------------add f1
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    #-------------------add f1 end
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask

        # ------------------fp16
        if use_fp16:
            with ac():
                loss = model((batch_data, batch_token_starts),
                        token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        # ------------------fp16 end
        else:
        # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                        token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        
        # ------------------fp16
        if use_fp16:
            scaler.scale(loss).backward()
        # ------------------fp16 end
        else:
            loss.backward()
        
        #---------------fgm
        if fgm is not None:
            fgm.attack()
            if use_fp16:
                with ac():
                    loss_adv = model((batch_data, batch_token_starts),
                            token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
            else:
                loss_adv = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
            if use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            fgm.restore()
        #---------------fgm end
        #---------------pgd
        elif pgd is not None:
            pgd.backup_grad()
            for _t in range(pgd_k):
                pgd.attack(is_first_attack=(_t == 0))

                if _t != pgd_k - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                if use_fp16:
                    with ac():
                        loss_adv = model((batch_data, batch_token_starts),
                            token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
                else:
                    loss_adv = model((batch_data, batch_token_starts),
                        token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]   
                if use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
            pgd.restore()
        #---------------pgd end
        #---------------FP16
        if use_fp16:
            scaler.unscale_(optimizer)
        
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        #---------------add f1
        label_masks = batch_labels.gt(-1)
        batch_output = model((batch_data, batch_token_starts),
                        token_type_ids=None, attention_mask=batch_masks)[0]
        batch_output = model.crf.decode(batch_output, mask=label_masks)
        batch_labels = batch_labels.to('cpu').numpy()
        pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
        true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_labels])
        #---------------add f1 end
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        
    #---------------add f1
    assert len(pred_tags) == len(true_tags)
    f1 = f1_score(true_tags, pred_tags, 'dev')
    #---------------add f1 end
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}, f1 score: {}".format(epoch, train_loss, f1))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        #-------------------modified
        # model = CRFModel(bert_dir=config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            #------------------modified
            # output_dir = os.path.join(model_dir, 'checkpoint-{}'.format(epoch))
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir, exist_ok=True)

            # # take care of model distributed / parallel training
            # model_to_save = (
            #     model.module if hasattr(model, "module") else model
            # )
            # torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
