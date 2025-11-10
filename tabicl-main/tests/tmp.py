from tabicl.prior.data_reader.data_reader import DataReader
dr = DataReader(UEA_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",  UCR_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",transform_ts_size=512)
X1, y1 = dr.read_dataset("InsectWingbeat", which_set="train")
X2, y2 = dr.read_dataset("InsectWingbeat", which_set="train")
assert X1.shape[0] == 1000
assert (X1 == X2).all()  # deterministic for same DataReader parameters