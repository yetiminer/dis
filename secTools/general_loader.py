from secTools import yamlLoad, load_from_db, streamline
from load_data1 import SecTable, SECdataset
from sqlalchemy import create_engine, MetaData

def ds_from_db(**kwargs):

	cfg=kwargs['cfg']
	db_name=kwargs['db_name']
	db_location=kwargs['db_location']
	
	engine=create_engine('sqlite:///'+db_location, echo=True)
	data=load_from_db(engine)
	data=streamline(data,cfg)
	ds=SECdataset('ds',cfg,**data)
	return ds
	
	
