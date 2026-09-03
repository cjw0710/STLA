import os
import time
import logging

import torch
from utils.Setup import setup,trans_to_cuda
from utils.Optim import build_optimizer
from utils.Metric import Metrics
from dataLoader import create_dataloaders
from config import parse_args
from model import DeDiff,loss_function
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, add_self_loops,from_scipy_sparse_matrix,softmax
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from graph import get_info

def train(args):
    global best_scores
    train_dataloader, val_dataloader, test_dataloader , train_dataset= create_dataloaders(args)
    info  = get_info(args, train_dataloader, train_dataset)
    model = DeDiff(args)
    model = model.to(args.device)
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    step = 0
    best_score = args.best_score  
    start_time = time.time()

    for epoch in range(args.max_epochs):
        print('\n[ Training Epoch ', epoch, ']')
        for batch in train_dataloader:
            model.train()  
            pred_user, label,process_data = model(args, batch,info)
            loss = loss_function(pred_user, label,process_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")
        t_scores = inference(args, model, test_dataloader,info)
        print(' # ----------Test Result---------')
        for metric in t_scores.keys():
            print(' ' + metric + ' ' + str(t_scores[metric]))
        if sum(t_scores.values()) > best_score:
            best_score = sum(t_scores.values())
            best_scores = t_scores
            torch.save({'model_state_dict': model.state_dict()}, f'{args.saved_model_path}/{args.dataset}.bin')
            print(' --> Save Model <-- ')

    print('\n #-------Reported Result-------')
    for metric in best_scores.keys():
        print(' ' + metric + ' ' + str(best_scores[metric]))

def inference(args, model, dataloader,info):

    model.eval()  
    k_list = args.metric_k  
    scores = {}
    for k in k_list:
        scores['hit@' + str(k)] = 0  
        scores['map@' + str(k)] = 0  
    
    n_total_words = 0  
    with torch.no_grad():
        for batch in dataloader:
            prediction, label ,_= model(args, batch,info)
            scores_batch, scores_len = Metrics(args).compute_metric(prediction, label.contiguous().view(-1))
            n_total_words += scores_len
            for k in k_list:
                scores['hit@' + str(k)] += scores_batch['hit@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
    for k in k_list:
        scores['hit@' + str(k)] = scores['hit@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores

def main():
    args = parse_args()
    setup(args)
    os.makedirs(args.saved_model_path, exist_ok=True)
    logging.info("Training/Testing parameters: %s", args)
    train(args)

if __name__ == '__main__':
    main()
