from predictability_emergence.lonlat_fromcsv import lons_lats

def test_1(data_dir):
    if data_dir is not None:
        lonvec, latvec = lons_lats(f'{data_dir}/test_inputs.csv')
        assert len(lonvec) == 2
        assert len(latvec) == 1

def test_2(data_dir):
    if data_dir is not None:
        lonvec, latvec = lons_lats(f'{data_dir}/inputs.csv')
        assert len(lonvec) == 36
        assert len(latvec) == 1
