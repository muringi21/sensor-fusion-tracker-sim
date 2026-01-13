def test_import_and_step():
    import tracker_core
    mt = tracker_core.MultiTrackerCV(dt=0.1)
    out = mt.step([(0.0, 0.0), (10.0, 0.0)])
    assert isinstance(out, list)