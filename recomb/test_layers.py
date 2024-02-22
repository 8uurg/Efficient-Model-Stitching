import layers
import torch

def test_relu_find_feature_change_start_from_end():
    l = layers.ReLU()
    assert l.find_feature_change_start(l.point_end()) == None

def test_linear_find_feature_change_start_from_end():
    l = layers.Linear(10, 10)
    # This layer would be able to change the start of this feature map span -
    # as this span is started by this node's output
    assert l.find_feature_change_start(l.point_end()) == l.point_end()

def test_linear_find_feature_change_start_from_start():
    l = layers.Linear(10, 10)
    # Not the case here: input != output
    assert l.find_feature_change_start(l.point_start()) is None

def test_sequential_w_change_find_feature_change_start_from_end():
    l = layers.Sequential(
        layers.Linear(10, 10),
        layers.ReLU()
    )
    p = l.find_feature_change_start(l.point_end())
    assert p[1] == 0

def test_sequential_w_change_find_feature_change_end_from_start():
    l = layers.Sequential(
        layers.Linear(10, 10),
        layers.ReLU()
    )
    p = l.find_feature_change_end(l.point_start())
    assert p[1] == 0


def test_sequential_w_o_change_find_feature_change_start_from_end():
    l = layers.Sequential(
        layers.ReLU(),
        layers.ReLU()
    )
    p = l.find_feature_change_start(l.point_end())
    assert p is None

def test_sequential_w_o_change_find_feature_change_end_from_start():
    l = layers.Sequential(
        layers.ReLU(),
        layers.ReLU()
    )
    p = l.find_feature_change_end(l.point_start())
    assert p is None

def test_points_convnet():
    l = layers.get_simple_convnet(1, 10)
    for p in l.enumerate_points():
        s = l.find_feature_change_start(p)
        e = l.find_feature_change_end(p)
        if s is None:
            assert e is not None
        elif e is None:
            assert s is not None
        else:
            assert s[1] < e[1]

def test_split_relu_before():
    l = layers.ReLU()
    
    s = l.split_network(None)
    x = torch.tensor([-1.0, 1.0])
    x_after = torch.tensor([0.0, 1.0])
    x_, f = s.prefix(x)
    assert (x_ == x).all()
    assert f is None
    f = s.parallel(f)
    assert f is None
    x_ = s.suffix((x_, f))
    assert (x_ == x_after).all()

def test_split_relu_after():
    l = layers.ReLU()
    
    s = l.split_network(l)
    x = torch.tensor([-1.0, 1.0])
    x_after = torch.tensor([0.0, 1.0])
    x_, f = s.prefix(x)
    assert (x_ == x_after).all()
    assert f is None
    f = s.parallel(f)
    assert f is None
    x_ = s.suffix((x_, f))
    assert (x_ == x_after).all()


def test_split_sequential_before():
    l = layers.Sequential(
        layers.Linear(2, 4),
        layers.ReLU()
    )
    
    s = l.split_network(None)
    x = torch.tensor([-1.0, 1.0])
    x_after = l(x)
    x_, f = s.prefix(x)
    assert (x_ == x).all()
    assert f is None
    f = s.parallel(f)
    assert f is None
    x_ = s.suffix((x_, f))
    assert (x_ == x_after).all()

def test_split_sequential_between():
    linear = layers.Linear(2, 4)
    relu = layers.ReLU()
    l = layers.Sequential(
        linear, relu
    )
    
    s = l.split_network((l, 0, linear))
    x = torch.tensor([-1.0, 1.0])
    x_after = l(x)
    x_, f = s.prefix(x)
    assert (x_ == linear(x)).all()
    assert f is None
    f = s.parallel(f)
    assert f is None
    x_ = s.suffix((x_, f))
    assert (x_ == x_after).all()


def test_split_concatenate_before():
    l = layers.Concatenate(
        layers.Linear(2, 4),
        layers.ReLU()
    )
    
    s = l.split_network(None)
    x = torch.tensor([-1.0, 1.0])
    x_after = l(x)
    x_, f = s.prefix(x)
    assert (x_ == x).all()
    assert f is None
    f = s.parallel(f)
    assert f is None
    x_ = s.suffix(x_, f)
    assert (x_ == x_after).all()

def test_split_concatenate_between():
    linear = layers.Linear(2, 4)
    relu = layers.ReLU()
    l = layers.Concatenate(
        linear, relu
    )
    
    s = l.split_network((l, 0, linear))
    x = torch.tensor([-1.0, 1.0])
    x_after = l(x)
    x_, f = s.prefix(x)
    assert (x_ == linear(x)).all()
    assert f[0] is None
    assert (f[1] == x).all()
    f = s.parallel(f)
    assert f[0] is None
    assert len(f[1]) == 2, "Concatenate stores two lists of vectors - before idx and after idx"
    assert len(f[1][0]) == 0, "Splitting on the first index - prefix should be empty"
    assert len(f[1][1]) == 1, "Splitting on the first index for two elements - suffix should be of size one"
    assert (f[1][1][0] == relu(x)).all()
    x_ = s.suffix((x_, f))
    assert (x_ == x_after).all()

def test_reinit_linear_default():
    l = layers.Linear(4, 10)
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    o_o = l(t)

    l.reinit()
    o_a = l(t)
    assert (o_o != o_a).all()

def test_reinit_linear():
    l = layers.Linear(4, 10)
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    o_o = l(t)

    l.reinit(None, None)
    o_a = l(t)
    assert (o_o == o_a).all()

def test_reinit_sequential_default():
    l = layers.Sequential(
        layers.Linear(4, 10),
    )
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    o_o = l(t)

    l.reinit()
    o_a = l(t)
    assert (o_o != o_a).all()

def test_reinit_sequential_simple():
    l = layers.Sequential(
        layers.Linear(4, 10),
    )
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    o_o = l(t)

    l.reinit(None, None)
    o_a = l(t)
    assert (o_o == o_a).all()

def test_reinit_sequential_1():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit(None, None)

    assert (l_a(t_in) == r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_sequential_2():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit(None, (l, 0, l_a))

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_sequential_3():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit(None, (l, 1, l_b))

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) != r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_sequential_3():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit(None, (l, 2, l_c))

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) != r_b).all()
    assert (l_c(t_in) != r_c).all()

def test_reinit_sequential_4():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit((l, 0, l_a), (l, 2, l_c))

    assert (l_a(t_in) == r_a).all()
    assert (l_b(t_in) != r_b).all()
    assert (l_c(t_in) != r_c).all()

def test_reinit_sequential_5():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit((l, 1, l_b), (l, 2, l_c))

    assert (l_a(t_in) == r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) != r_c).all()

def test_reinit_sequential_6():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Sequential(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit((l, 2, l_c), (l, 2, l_c))

    assert (l_a(t_in) == r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_concat_0():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Concatenate(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    l.reinit(None, None)

    assert (l_a(t_in) == r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_concat_1():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Concatenate(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    # Full reinit
    l.reinit(None, l.point_end())

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) != r_b).all()
    assert (l_c(t_in) != r_c).all()

def test_reinit_concat_2():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Concatenate(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    # Spillover from input
    l.reinit(None, (l, 0, l_a))

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_concat_2():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Concatenate(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    # Spillover from output
    l.reinit((l, 0, None), l.point_end())

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()

def test_reinit_concat_3():
    l_a = layers.Linear(4, 4)
    l_b = layers.Linear(4, 4)
    l_c = layers.Linear(4, 4)

    l = layers.Concatenate(
        l_a, l_b, l_c
    )

    t_in = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r_a = l_a(t_in)
    r_b = l_b(t_in)
    r_c = l_c(t_in)

    # Node restricted
    l.reinit((l, 0, None), (l, 0, l_a))

    assert (l_a(t_in) != r_a).all()
    assert (l_b(t_in) == r_b).all()
    assert (l_c(t_in) == r_c).all()