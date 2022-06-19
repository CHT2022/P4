from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import dummy
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import KFold
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder





app = Flask(__name__)


#Welcome Page
@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method=="POST":
        return render_template('result.html')
    return render_template('index.html')

#Results Page
@app.route("/result", methods=["GET", "POST"])
def result():

	if request.method == 'POST':
		# Import dataset 
		df = pd.read_csv('data2016.csv')
		df = df.drop(columns=['Unnamed: 0'], axis=1)

		# One hot encoding
		df = pd.get_dummies(df, columns = ['DEST'])
		df = pd.get_dummies(df, columns = ['CARRIER'])
		df = pd.get_dummies(df, columns = ['MONTH'])
		
		

		# Label Encoding
		le_origin = LabelEncoder()
		df['ORIGIN'] = le_origin.fit_transform(df['ORIGIN'])

		X = df.drop('ARR_DELAY',axis=1)
		y = df.ARR_DELAY

		rr = Ridge(alpha=1.6)
		rr.fit(X, y)

		month = request.form['hmonth']
		carrier = request.form['hcarrier']
		origin = request.form['horigin']
		dest = request.form['hdest']
		dep_delay= request.form['hdep_delay']
		


	def predict_delay(origin,destination,carrier,month,dep_delay):
		X1 = [{'ORIGIN':origin,
            'DEST_IND':1 if destination=='IND' else 0,
            'DEST_ORD':1 if destination=='ORD' else 0,
            'DEST_SFO':1 if destination=='SFO' else 0,
            'DEST_JAX':1 if destination=='JAX' else 0,
            'DEST_SNA':1 if destination=='SNA' else 0,
            'DEST_IAH':1 if destination=='IAH' else 0,
            'DEST_PHX':1 if destination=='PHX' else 0,
            'DEST_BNA':1 if destination=='BNA' else 0,
            'DEST_LGA':1 if destination=='LGA' else 0,
            'DEST_EWR':1 if destination=='EWR' else 0,
            'DEST_GEG':1 if destination=='GEZ' else 0,
            'DEST_AUS':1 if destination=='AUS' else 0,
            'DEST_MAF':1 if destination=='MAF' else 0,
            'DEST_GRR':1 if destination=='GRR' else 0,
            'DEST_SLC':1 if destination=='SLC' else 0,
            'DEST_MCO':1 if destination=='MCO' else 0,
            'DEST_MDW':1 if destination=='MDW' else 0,
            'DEST_DFW':1 if destination=='DFW' else 0,
            'DEST_MGM':1 if destination=='MGM' else 0,
            'DEST_PDX':1 if destination=='PDX' else 0,
            'DEST_FSM':1 if destination=='FSM' else 0,
            'DEST_ATL':1 if destination=='ATL' else 0,
            'DEST_PBI':1 if destination=='PBI' else 0,
            'DEST_LAS':1 if destination=='LAS' else 0,
            'DEST_TPA':1 if destination=='TPA' else 0,
            'DEST_MEM':1 if destination=='MEM' else 0,
            'DEST_MSP':1 if destination=='MSP' else 0,
            'DEST_LAX':1 if destination=='LAX' else 0,
            'DEST_SEA':1 if destination=='SEA' else 0,
            'DEST_HNL':1 if destination=='HNL' else 0,
            'DEST_DEN':1 if destination=='DEN' else 0,
            'DEST_BWI':1 if destination=='BWI' else 0,
            'DEST_SAT':1 if destination=='SAT' else 0,
            'DEST_PWM':1 if destination=='PWM' else 0,
            'DEST_CLT':1 if destination=='CLT' else 0,
            'DEST_CLE':1 if destination=='CLE' else 0,
            'DEST_JFK':1 if destination=='JFK' else 0,
            'DEST_FLL':1 if destination=='FLL' else 0,
            'DEST_MIA':1 if destination=='MIA' else 0,
            'DEST_BOS':1 if destination=='BOS' else 0,
            'DEST_TLH':1 if destination=='TLH' else 0,
            'DEST_SBA':1 if destination=='SBA' else 0,
            'DEST_BUF':1 if destination=='BUF' else 0,
            'DEST_PSP':1 if destination=='PSP' else 0,
            'DEST_OKC':1 if destination=='OKC' else 0,
            'DEST_OMA':1 if destination=='OMA' else 0,
            'DEST_HOU':1 if destination=='HOU' else 0,
            'DEST_CMH':1 if destination=='CMH' else 0,
            'DEST_DTW':1 if destination=='DTW' else 0,
            'DEST_RIC':1 if destination=='RIC' else 0,
            'DEST_BMI':1 if destination=='BMI' else 0,
            'DEST_PHL':1 if destination=='PHL' else 0,
            'DEST_BDL':1 if destination=='BDL' else 0,
            'DEST_ICT':1 if destination=='ICT' else 0,
            'DEST_PVD':1 if destination=='PVD' else 0,
            'DEST_IAD':1 if destination=='IAD' else 0,
            'DEST_SMF':1 if destination=='SMF' else 0,
            'DEST_ORF':1 if destination=='ORF' else 0,
            'DEST_OAK':1 if destination=='OAK' else 0,
            'DEST_DLH':1 if destination=='DLH' else 0,
            'DEST_LIT':1 if destination=='LIT' else 0,
            'DEST_ANC':1 if destination=='ANC' else 0,
            'DEST_RSW':1 if destination=='RSW' else 0,
            'DEST_ONT':1 if destination=='ONT' else 0,
            'DEST_DAY':1 if destination=='DAY' else 0,
            'DEST_STL':1 if destination=='STL' else 0,
            'DEST_BTR':1 if destination=='BTR' else 0,
            'DEST_HSV':1 if destination=='HSV' else 0,
            'DEST_MSY':1 if destination=='MSY' else 0,
            'DEST_BOI':1 if destination=='BOI' else 0,
            'DEST_DCA':1 if destination=='DCA' else 0,
            'DEST_TTN':1 if destination=='TTN' else 0,
            'DEST_GNV':1 if destination=='GNV' else 0,
            'DEST_ITO':1 if destination=='ITO' else 0,
            'DEST_SAN':1 if destination=='SAN' else 0,
            'DEST_TUS':1 if destination=='TUS' else 0,
            'DEST_KOA':1 if destination=='KOA' else 0,
            'DEST_GSP':1 if destination=='GSP' else 0,
            'DEST_MFE':1 if destination=='MFE' else 0,
            'DEST_JAN':1 if destination=='JAN' else 0,
            'DEST_DAL':1 if destination=='DAL' else 0,
            'DEST_OGG':1 if destination=='OGG' else 0,
            'DEST_SDF':1 if destination=='SDF' else 0,
            'DEST_LIH':1 if destination=='LIH' else 0,
            'DEST_BHM':1 if destination=='BHM' else 0,
            'DEST_PIT':1 if destination=='PIT' else 0,
            'DEST_EUG':1 if destination=='EUG' else 0,
            'DEST_ISP':1 if destination=='ISP' else 0,
            'DEST_TUL':1 if destination=='TUL' else 0,
            'DEST_LBB':1 if destination=='LBB' else 0,
            'DEST_CHA':1 if destination=='CHA' else 0,
            'DEST_MHT':1 if destination=='MHT' else 0,
            'DEST_RDU':1 if destination=='RDU' else 0,
            'DEST_LEX':1 if destination=='LEX' else 0,
            'DEST_ABQ':1 if destination=='ABQ' else 0,
            'DEST_SGF':1 if destination=='SGF' else 0,
            'DEST_CHS':1 if destination=='CHS' else 0,
            'DEST_MKE':1 if destination=='MKE' else 0,
            'DEST_SJU':1 if destination=='SJU' else 0,
            'DEST_EGE':1 if destination=='EGE' else 0,
            'DEST_BFL':1 if destination=='BFL' else 0,
            'DEST_MCI':1 if destination=='MCI' else 0,
            'DEST_ILM':1 if destination=='ILM' else 0,
            'DEST_STT':1 if destination=='STT' else 0,
            'DEST_EAU':1 if destination=='EAU' else 0,
            'DEST_PLN':1 if destination=='PLN' else 0,
            'DEST_MLI':1 if destination=='MLI' else 0,
            'DEST_FLG':1 if destination=='FLG' else 0,
            'DEST_PSC':1 if destination=='PSC' else 0,
            'DEST_PIA':1 if destination=='PIA' else 0,
            'DEST_MSN':1 if destination=='MSN' else 0,
            'DEST_CRP':1 if destination=='CRP' else 0,
            'DEST_SJC':1 if destination=='SJC' else 0,
            'DEST_FAR':1 if destination=='FAR' else 0,
            'DEST_MLU':1 if destination=='MLU' else 0,
            'DEST_BQK':1 if destination=='BQK' else 0,
            'DEST_EYW':1 if destination=='EYW' else 0,
            'DEST_CPR':1 if destination=='CPR' else 0,
            'DEST_TYR':1 if destination=='TYR' else 0,
            'DEST_BTV':1 if destination=='BTV' else 0,
            'DEST_GPT':1 if destination=='GPT' else 0,
            'DEST_RNO':1 if destination=='RNO' else 0,
            'DEST_ATW':1 if destination=='ATW' else 0,
            'DEST_FAI':1 if destination=='FAI' else 0,
            'DEST_FAT':1 if destination=='FAT' else 0,
            'DEST_LGB':1 if destination=='LGB' else 0,
            'DEST_COS':1 if destination=='COS' else 0,
            'DEST_FWA':1 if destination=='FWA' else 0,
            'DEST_CVG':1 if destination=='CVG' else 0,
            'DEST_ALB':1 if destination=='ALB' else 0,
            'DEST_AEX':1 if destination=='AEX' else 0,
            'DEST_ACV':1 if destination=='ACV' else 0,
            'DEST_ELM':1 if destination=='ELM' else 0,
            'DEST_ELP':1 if destination=='ELP' else 0,
            'DEST_SAV':1 if destination=='SAV' else 0,
            'DEST_ABE':1 if destination=='ABE' else 0,
            'DEST_XNA':1 if destination=='XNA' else 0,
            'DEST_FSD':1 if destination=='FSD' else 0,
            'DEST_BUR':1 if destination=='BUR' else 0,
            'DEST_SIT':1 if destination=='SIT' else 0,
            'DEST_TYS':1 if destination=='TYS' else 0,
            'DEST_HRL':1 if destination=='HRL' else 0,
            'DEST_SRQ':1 if destination=='SRQ' else 0,
            'DEST_SBN':1 if destination=='SBN' else 0,
            'DEST_LWS':1 if destination=='LWS' else 0,
            'DEST_GTF':1 if destination=='GTF' else 0,
            'DEST_ECP':1 if destination=='ECP' else 0,
            'DEST_JAC':1 if destination=='JAC' else 0,
            'DEST_RAP':1 if destination=='RAP' else 0,
            'DEST_ISN':1 if destination=='ISN' else 0,
            'DEST_MFR':1 if destination=='MFR' else 0,
            'DEST_GCC':1 if destination=='GCC' else 0,
            'DEST_MYR':1 if destination=='MYR' else 0,
            'DEST_HPN':1 if destination=='HPN' else 0,
            'DEST_BRO':1 if destination=='BRO' else 0,
            'DEST_AGS':1 if destination=='AGS' else 0,
            'DEST_AMA':1 if destination=='AMA' else 0,
            'DEST_JNU':1 if destination=='JNU' else 0,
            'DEST_PNS':1 if destination=='PNS' else 0,
            'DEST_TRI':1 if destination=='TRI' else 0,
            'DEST_FCA':1 if destination=='FCA' else 0,
            'DEST_SYR':1 if destination=='SYR' else 0,
            'DEST_IDA':1 if destination=='IDA' else 0,
            'DEST_SBP':1 if destination=='SBP' else 0,
            'DEST_SMX':1 if destination=='SMX' else 0,
            'DEST_SJT':1 if destination=='SJT' else 0,
            'DEST_BGR':1 if destination=='BGR' else 0,
            'DEST_MSO':1 if destination=='MSO' else 0,
            'DEST_LRD':1 if destination=='LRD' else 0,
            'DEST_SGU':1 if destination=='SGU' else 0,
            'DEST_CAK':1 if destination=='CAK' else 0,
            'DEST_SPI':1 if destination=='SPI' else 0,
            'DEST_GRB':1 if destination=='GRB' else 0,
            'DEST_GSO':1 if destination=='GSO' else 0,
            'DEST_RST':1 if destination=='RST' else 0,
            'DEST_DSM':1 if destination=='DSM' else 0,
            'DEST_AVL':1 if destination=='AVL' else 0,
            'DEST_DHN':1 if destination=='DHN' else 0,
            'DEST_AZO':1 if destination=='AZO' else 0,
            'DEST_CLL':1 if destination=='CLL' else 0,
            'DEST_BRD':1 if destination=='BRD' else 0,
            'DEST_MMH':1 if destination=='MMH' else 0,
            'DEST_BIS':1 if destination=='BIS' else 0,
            'DEST_ITH':1 if destination=='ITH' else 0,
            'DEST_MLB':1 if destination=='MLB' else 0,
            'DEST_BZN':1 if destination=='BZN' else 0,
            'DEST_OTZ':1 if destination=='OTZ' else 0,
            'DEST_RDM':1 if destination=='RDM' else 0,
            'DEST_BRW':1 if destination=='BRW' else 0,
            'DEST_LFT':1 if destination=='LFT' else 0,
            'DEST_KTN':1 if destination=='KTN' else 0,
            'DEST_ROC':1 if destination=='ROC' else 0,
            'DEST_VPS':1 if destination=='VPS' else 0,
            'DEST_EKO':1 if destination=='EKO' else 0,
            'DEST_PSE':1 if destination=='PSE' else 0,
            'DEST_CID':1 if destination=='CID' else 0,
            'DEST_HDN':1 if destination=='HDN' else 0,
            'DEST_AVP':1 if destination=='AVP' else 0,
            'DEST_LAN':1 if destination=='LAN' else 0,
            'DEST_BIL':1 if destination=='BIL' else 0,
            'DEST_PGD':1 if destination=='PGD' else 0,
            'DEST_RKS':1 if destination=='RKS' else 0,
            'DEST_CIU':1 if destination=='CIU' else 0,
            'DEST_FNT':1 if destination=='FNT' else 0,
            'DEST_SHV':1 if destination=='SHV' else 0,
            'DEST_YUM':1 if destination=='YUM' else 0,
            'DEST_OME':1 if destination=='OME' else 0,
            'DEST_ASE':1 if destination=='ASE' else 0,
            'DEST_DRO':1 if destination=='DRO' else 0,
            'DEST_TXK':1 if destination=='TXK' else 0,
            'DEST_ACT':1 if destination=='ACT' else 0,
            'DEST_ROA':1 if destination=='ROA' else 0,
            'DEST_APN':1 if destination=='APN' else 0,
            'DEST_GCK':1 if destination=='GCK' else 0,
            'DEST_ORH':1 if destination=='ORH' else 0,
            'DEST_CDV':1 if destination=='CDV' else 0,
            'DEST_ACY':1 if destination=='ACY' else 0,
            'DEST_PHF':1 if destination=='PHF' else 0,
            'DEST_HYS':1 if destination=='HYS' else 0,
            'DEST_SAF':1 if destination=='SAF' else 0,
            'DEST_GFK':1 if destination=='GFK' else 0,
            'DEST_CAE':1 if destination=='CAE' else 0,
            'DEST_SCE':1 if destination=='SCE' else 0,
            'DEST_COD':1 if destination=='COD' else 0,
            'DEST_GRK':1 if destination=='GRK' else 0,
            'DEST_SUN':1 if destination=='SUN' else 0,
            'DEST_LNK':1 if destination=='LNK' else 0,
            'DEST_CWA':1 if destination=='CWA' else 0,
            'DEST_MOT':1 if destination=='MOT' else 0,
            'DEST_EVV':1 if destination=='EVV' else 0,
            'DEST_STX':1 if destination=='STX' else 0,
            'DEST_RDD':1 if destination=='RDD' else 0,
            'DEST_ADQ':1 if destination=='ADQ' else 0,
            'DEST_GTR':1 if destination=='GTR' else 0,
            'DEST_IMT':1 if destination=='IMT' else 0,
            'DEST_OAJ':1 if destination=='OAJ' else 0,
            'DEST_DVL':1 if destination=='DVL' else 0,
            'DEST_LAW':1 if destination=='LAW' else 0,
            'DEST_YAK':1 if destination=='YAK' else 0,
            'DEST_ABY':1 if destination=='ABY' else 0,
            'DEST_GJT':1 if destination=='GJT' else 0,
            'DEST_TWF':1 if destination=='TWF' else 0,
            'DEST_HLN':1 if destination=='HLN' else 0,
            'DEST_ACK':1 if destination=='ACK' else 0,
            'DEST_MKG':1 if destination=='MKG' else 0,
            'DEST_BPT':1 if destination=='BPT' else 0,
            'DEST_DAB':1 if destination=='DAB' else 0,
            'DEST_MTJ':1 if destination=='MTJ' else 0,
            'DEST_PIH':1 if destination=='PIH' else 0,
            'DEST_MEI':1 if destination=='MEI' else 0,
            'DEST_CRW':1 if destination=='CRW' else 0,
            'DEST_ERI':1 if destination=='ERI' else 0,
            'DEST_MRY':1 if destination=='MRY' else 0,
            'DEST_MOB':1 if destination=='MOB' else 0,
            'DEST_RHI':1 if destination=='RHI' else 0,
            'DEST_FAY':1 if destination=='FAY' else 0,
            'DEST_TVC':1 if destination=='TVC' else 0,
            'DEST_SWF':1 if destination=='SWF' else 0,
            'DEST_LCH':1 if destination=='LCH' else 0,
            'DEST_VLD':1 if destination=='VLD' else 0,
            'DEST_BQN':1 if destination=='BQN' else 0,
            'DEST_MBS':1 if destination=='MBS' else 0,
            'DEST_JMS':1 if destination=='JMS' else 0,
            'DEST_MQT':1 if destination=='MQT' else 0,
            'DEST_PSG':1 if destination=='PSG' else 0,
            'DEST_SCC':1 if destination=='SCC' else 0,
            'DEST_CHO':1 if destination=='CHO' else 0,
            'DEST_MDT':1 if destination=='MDT' else 0,
            'DEST_PIB':1 if destination=='PIB' else 0,
            'DEST_CSG':1 if destination=='CSG' else 0,
            'DEST_ESC':1 if destination=='ESC' else 0,
            'DEST_CMX':1 if destination=='CMX' else 0,
            'DEST_HOB':1 if destination=='HOB' else 0,
            'DEST_BJI':1 if destination=='BJI' else 0,
            'DEST_EWN':1 if destination=='EWN' else 0,
            'DEST_LAR':1 if destination=='LAR' else 0,
            'DEST_MVY':1 if destination=='MVY' else 0,
            'DEST_BET':1 if destination=='BET' else 0,
            'DEST_INL':1 if destination=='INL' else 0,
            'DEST_ABR':1 if destination=='ABR' else 0,
            'DEST_WRG':1 if destination=='WRG' else 0,
            'DEST_BGM':1 if destination=='BGM' else 0,
            'DEST_UST':1 if destination=='UST' else 0,
            'DEST_CDC':1 if destination=='CDC' else 0,
            'DEST_PAH':1 if destination=='PAH' else 0,
            'DEST_JLN':1 if destination=='JLN' else 0,
            'DEST_SPS':1 if destination=='SPS' else 0,
            'DEST_AKN':1 if destination=='AKN' else 0,
            'DEST_IAG':1 if destination=='IAG' else 0,
            'DEST_LBE':1 if destination=='LBE' else 0,
            'DEST_GUM':1 if destination=='GUM' else 0,
            'DEST_GRI':1 if destination=='GRI' else 0,
            'DEST_WYS':1 if destination=='WYS' else 0,
            'DEST_BLI':1 if destination=='BLI' else 0,
            'DEST_GGG':1 if destination=='GGG' else 0,
            'DEST_ROW':1 if destination=='ROW' else 0,
            'DEST_HIB':1 if destination=='HIB' else 0,
            'DEST_GUC':1 if destination=='GUC' else 0,
            'DEST_LSE':1 if destination=='LSE' else 0,
            'DEST_BTM':1 if destination=='BTM' else 0,
            'DEST_PBG':1 if destination=='PBG' else 0,
            'DEST_ABI':1 if destination=='ABI' else 0,
            'DEST_ADK':1 if destination=='ADK' else 0,
            'DEST_PPG':1 if destination=='PPG' else 0,
            'DEST_GST':1 if destination=='GST' else 0,
            'DEST_DLG':1 if destination=='DLG' else 0,           
            'CARRIER_Delta Airlines':1 if carrier == 'Delta Airlines' else 0,
            'CARRIER_American Airlines':1 if carrier == 'American Airlines' else 0,
            'CARRIER_Alaska Airlines':1 if carrier == 'Alaska Airlines' else 0,
            'CARRIER_JetBlue Airways':1 if carrier == 'JetBlue Airways' else 0,
            'CARRIER_ExpressJet':1 if carrier == 'ExpressJet' else 0,
            'CARRIER_Frontier Airlines':1 if carrier == 'Frontier Airlines' else 0,
            'CARRIER_Virgin America':1 if carrier == 'Virgin America' else 0,
            'CARRIER_Southwest Airlines':1 if carrier == 'Southwest Airlines' else 0,
            'CARRIER_SkyWest Airlines':1 if carrier == 'SkyWest Airlines' else 0,
            'CARRIER_United Airlines':1 if carrier == 'United Airlines' else 0,
            'CARRIER_Hawaiian Airlines':1 if carrier == 'Hawaiian Airlines' else 0,
            'CARRIER_Spirit Airlines':1 if carrier == 'Spirit Airlines' else 0,
            'MONTH_1':1 if month == 1 else 0,
            'MONTH_2':1 if month == 2 else 0,
            'MONTH_3':1 if month == 3 else 0,
            'MONTH_4':1 if month == 4 else 0,
            'MONTH_5':1 if month == 5 else 0,
            'MONTH_6':1 if month == 6 else 0,
            'MONTH_7':1 if month == 7 else 0,
            'MONTH_8':1 if month == 8 else 0,
            'MONTH_9':1 if month == 9 else 0,
            'MONTH_10':1 if month == 10 else 0,
            'MONTH_11':1 if month == 11 else 0,
            'MONTH_12':1 if month == 12 else 0,
            'DEP_DELAY': dep_delay}]  
		    
		X2 = pd.DataFrame(X1)
            
		label = preprocessing.LabelEncoder()
		X2['ORIGIN'] = label.fit_transform(X2['ORIGIN'])

		


		pred_delay = rr.predict(pd.DataFrame(X2))
		return int(pred_delay[0])

	try:
		output = predict_delay(origin,dest,carrier,month,dep_delay)
		return render_template('result.html', output=output)
	except ValueError as e:
		return render_template('index.html', error=e) 
    

    
if __name__ == '__main__':
	app.run(debug= True)
