# PSUEDO

# STEP 1

def forward_pass(x,h_prev):
    input x < - first word

    a = W@h_prev + U@x + b
    h = tanh(a)

    save a ?
    save h ?

    o = V@h + c

    p = softMax(o)

    new_char = sample_char(char_dict,p)

    y_1 = onehot vector for x_1

    l_1 = computeLoss(p_1,y_1)

    L += l_1

    return new_char , h

def forward prop(x)
    # in first step h_prev is zero vector(?)

    new_char , new_h = forward_pass(first_char,h_0)

    for word in book[1:end]:
        new_char , new_h = forward_pass(new_char,new_h)

def backward_pass(err):


# STEP 2
