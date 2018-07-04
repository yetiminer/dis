import pandas as pd
import numpy as np

def ratio_fin(df,first_field,second_field):

    
    f=(df[first_field]/df[second_field]).replace([np.inf, -np.inf,0], np.nan).dropna()
    
    print(len(f))
    return f
def drop_inf0nan(series):
    f=series.replace([np.inf, -np.inf,0], np.nan).dropna()
    print(len(f))
    return f

def make_fin_measure_dic():	
	u_fin_ratios={}
	u_fin_ratios['u_ROE']=['NetIncomeLoss','StockholdersEquity']
	u_fin_ratios['u_ROA']=['NetIncomeLoss','Assets']
	u_fin_ratios['u_ROA']=['NetIncomeLoss','Revenues']
	u_fin_ratios['u_EBITDA_margin']=['IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
									'Revenues']
	u_fin_ratios['u_Leverage']=['Assets','StockholdersEquity']
	u_fin_ratios['u_cf_to_income']=['NetCashProvidedByUsedInOperatingActivities','NetIncomeLoss']
	u_fin_ratios['u_asset_turns']=['Revenues','Assets']
	u_fin_ratios['u_profit_margin']=['NetIncomeLoss','Revenues']
	u_fin_ratios['u_profit_to_comp_income']=['NetIncomeLoss','ComprehensiveIncomeNetOfTax']
	u_fin_ratios['u_current_assets_to_TAssets']=['AssetsCurrent','Assets']
	u_fin_ratios['u_PPE_to_TAssets']=['PropertyPlantAndEquipmentNet','Assets']
	u_fin_ratios['u_Inventory_to_Rev']=['InventoryNet','Revenues']
	u_fin_ratios['u_Inventory_to_Assets']=['InventoryNet','Assets']
	u_fin_ratios['u_AR_to_Assets']=['AccountsReceivableNetCurrent','Assets']
	u_fin_ratios['u_AR_to_Rev']=['AccountsReceivableNetCurrent','Revenues']
	u_fin_ratios['u_interest_cov']=['InterestPaid','GrossProfit']
	u_fin_ratios['u_doubtful_to_AC']=['AllowanceForDoubtfulAccountsReceivableCurrent','AccountsReceivableNetCurrent'] #gross or net?
	u_fin_ratios['u_payables_to_inventories']=['AccountsPayableCurrent','InventoryNet']
	u_fin_ratios['u_accounts_payable_turnover']=['AccountsPayableCurrent','CostOfGoodsSold']
	u_fin_ratios['u_goodwill_to_assets']=['Goodwill','Assets']
	#u_fin_ratios['u_tax_rate']=
	return u_fin_ratios

def make_measures(u_fin_ratios,df1):
	u_fin={}
	for k in u_fin_ratios:
		print(k)
		u_fin[k]=ratio_fin(df1,u_fin_ratios[k][0],u_fin_ratios[k][1])
	 
	u_fin['u_tangible_assets_to_assets']=drop_inf0nan(u_fin['u_current_assets_to_TAssets']+u_fin['u_PPE_to_TAssets'])
	return pd.DataFrame(u_fin)
	
def cond_fields(df,field):
	s=df1[df1[field].notnull()].count().sort_values(ascending=False)
	display(s[s>1000])