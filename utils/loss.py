import paddle
import paddle.nn.functional as F

def gather_2d_on_last_dim(
        tensor,
        index,
        shape
):
    #like gather_nd
    flattened_tensor = tensor.reshape((-1, tensor.shape[-1]))
    flattened_index = index.reshape((-1,))
    flattened_gathered_tensor = flattened_tensor[
        paddle.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.reshape(shape)

def get_entropy(
        logits
):
    probs=F.softmax(logits,axis=-1)
    log_pros=paddle.log(probs)
    entropy=paddle.mean(-log_pros*probs,axis=-1)

    return entropy

def soft_q_loss_with_sparse_rewards_2(
        logits,
        logits_,
        actions,
        rewards,
):
    """
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
    """
    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    
    V = logits.logsumexp(axis=-1)
    A = Q - V

    # Target outputs
    Q_ = paddle.zeros_like(Q)
    A_ = paddle.zeros_like(Q)
    V_ = logits_.logsumexp(axis=-1)
    Q_[:, :-1] = V_[:, 1:]
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # Terminal V-target is the last V-target before the episode ends
    terminal_V_ = V_[:,-1]
    Q_[:,-1] = rewards
    A_[:,-1] = rewards - terminal_V_

    raw_losses = F.mse_loss(A, A_)
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_,
        "H": get_entropy(logits),
        "H_": get_entropy(logits_),
    }

    return raw_losses, quantities_to_log

def soft_q_loss_with_sparse_rewards_3(
        logits,
        logits_,
        actions,
        rewards,
):

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(axis=-1)
    A = Q - V

    # Target outputs
    V_ = logits_.logsumexp(axis=-1)

    A2 =A.flip(axis=-1).cumsum(axis=-1).flip(axis=-1)

    raw_losses = F.mse_loss(
        A2, rewards.reshape((-1, 1)) - V_)

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
    }

    return raw_losses, quantities_to_log

def soft_q_loss_with_sparse_rewards(
        logits,
        logits_,
        actions,
        rewards,
):
    raw_loss_2,log_2=soft_q_loss_with_sparse_rewards_2(
        logits,
        logits_,
        actions,
        rewards,)
    
    raw_loss_3,log_3=soft_q_loss_with_sparse_rewards_3(
        logits,
        logits_,
        actions,
        rewards,)
    
    loss=(raw_loss_2+raw_loss_3)/2

    log={}
    log["loss"]=loss.cpu().tolist()
    for key, value  in log_2.items():
        #log[f"2_{key}"]=value
        log[f"2_{key}_max"]=value.max(axis=1).cpu().tolist()
        log[f"2_{key}_min"]=value.min(axis=1).cpu().tolist()
        log[f"2_{key}_mean"]=value.mean(axis=1).cpu().tolist()

    for key, value  in log_3.items():
        #log[f"3_{key}"]=value
        log[f"3_{key}_max"]=value.max(axis=1).cpu().tolist()
        log[f"3_{key}_min"]=value.min(axis=1).cpu().tolist()
        log[f"3_{key}_mean"]=value.mean(axis=1).cpu().tolist()

    return loss, log
