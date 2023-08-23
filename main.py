import streamlit as st
import os
from streamlit_option_menu import option_menu
from universitas import universitas
from fakultas import fakultas
from program_studi import programstudi
from gen_datvis import respon_rate, plot_overall, tingkat_perusahaan_univ, sektor_perusahaan_univ, salary_an, sebaran_univ, top_perusahaan_univ, jabatan_univ, status_pekerjaan_univ, frekuensi_melanjutkan_pendidikan, sumber_biaya_melanjutkan_pendidikan, sebaran_melanjutkan_pendidikan, frekuensi_tidak_bekerja, level_motivasi_tidak_bekerja, frekuensi_wiraswasta, sebaran_sektor_pekerjaan_wiraswasta, sebaran_kesesuaian_pendidikan_wiraswasta, sebaran_gaji_wiraswasta, sebaran_posisi_jabatan_wiraswasta, waktu_tunggu_wiraswasta, filtering_tempat_wiraswasta, filtering_pdloc_wiraswasta, tempat_wiraswasta, status_perusahaan_wiraswasta, top_perusahaan_wiraswasta, gen_pendapatan, gen_wkt_tunggu
from gen_datvis_fakultas import fktunggu, levelfirm, salaryrate, fakultas_tempat, plot_fakultas_jabatan, frekuensi_wiraswasta_fakultas, sebaran_sektor_pekerjaan_wiraswasta_fakultas, sebaran_kesesuaian_pendidikan_wiraswasta_fakultas, sebaran_gaji_wiraswasta_fakultas, sebaran_posisi_jabatan_wiraswasta_fakultas, waktu_tunggu_wiraswasta_fakultas, status_perusahaan_wiraswasta_fakultas, top_perusahaan_wiraswasta_fakultas, sebaran_pendapatan_wiraswasta_fakultas
from gen_datvis_prodi import prodi_tunggu, tingkat_prodi, sektor_prodi, pendapatan_prodi, prodi2, plot_prodi2, distribusih, frekuensi_wiraswasta_prodi, sebaran_sektor_pekerjaan_wiraswasta_prodi, sebaran_kesesuaian_pendidikan_wiraswasta_prodi, sebaran_gaji_wiraswasta_prodi, sebaran_posisi_jabatan_wiraswasta_prodi, filtering_pdloc_wiraswasta_prodi, tempat_wiraswasta_prodi, status_perusahaan_wiraswasta_prodi, top_perusahaan_wiraswasta_prodi
from gen_datvis_survey import plot_survey
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap, maskoceans
from math import ceil
import math
# from model_colab import plot_overall

def is_folder_empty(path):
   return len(os.listdir(path)) == 0

path = 'fig'
if is_folder_empty(path):
    # Import data & Preprocessing data
    data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1')
    datapop = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet2')
    survey = pd.read_excel('data/survey-pengguna-2022.xlsx', sheet_name='Master')
    pop_dict = dict(zip(datapop['Program Studi'], datapop['Populasi']))
    ##########################################
    populasirespond = {}
    for i,k in enumerate(datapop['Program Studi']):
        x = k
        y = data['Program Studi'].value_counts()[k]
        populasirespond[x] = y
    ##########################################
    pop_dict['D3 Manajemen Pemasaran']=163
    loc = data['Tempat Bekerja'].value_counts().keys()
    loc2 =np.unique(loc)
    data.loc[data['Kapan anda memulai bekerja'] < 0, 'Kapan anda memulai bekerja'] = 0
    col2 = data['Program Studi'].unique()
    dtfa =[np.array(data[data['Program Studi']==k]['Kapan anda memulai bekerja']) for k in col2]
    dtfa = [x[~np.isnan(x)] for x in dtfa]
    for x in dtfa:
        x[x <0] = 0
        x[x >15]=15
    wkt = dict(zip(col2, dtfa))
    cek = [round(list(data.isnull().sum())[i]/len(data),2) for i in range(len(list(data.isnull().sum())))]
    col =data.columns
    pmissing=np.unique(np.array(cek))
    npmissing=[cek.count(i) for i in pmissing]

    # General Information Tracer Study
    # set font style and the size of matplotlib plot
    font = {'family' : 'sans-serif',
            'weight' : 'ultralight',
            'size'   : 14}
    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 14})
    status = list(data['Status Anda saat ini (F8)'].unique())
    df1=data[data['Status Anda saat ini (F8)']==status[0]]
    df2=data[data['Status Anda saat ini (F8)']==status[1]] # df2 adalah yang berstatus 'Melanjutkan Pendidikan'
    df3=data[data['Status Anda saat ini (F8)']==status[2]] # df3 adalah yang berstatus 'Tidak Bekerja'
    df4=data[data['Status Anda saat ini (F8)']==status[3]]
    df1.loc[df1['Kapan anda memulai bekerja'] < 0, 'Kapan anda memulai bekerja'] = 0
    upbound = 30
    col = list(data.columns)
    # del data2
    data2 = df1.copy()
    data2["NIM"]=data2["NIM"].values.astype(str)
    prodi = list(data['Program Studi'].unique())
    for i,k in enumerate(prodi):
        uplimit = upbound
        ind=0
        idxv = data2[(data2['Kapan anda memulai bekerja'].isna()) & (data2['Program Studi']==k)].index
        idxv2 = data2[(data2['Kapan anda memulai bekerja'].notna()) & (data2['Program Studi']==k)].index
        rep = np.array(data2.loc[idxv2,'Kapan anda memulai bekerja'])
        # data2.loc[idxv,'Kapan anda memulai bekerja']=np.random.normal(np.quantile(rep, 0.5),np.std(rep))#np.random.uniform(np.quantile(rep, 0.25),np.quantile(rep, 0.75))
    while (data2[data2['Program Studi']==k]['Kapan anda memulai bekerja'].mean()>4): #& (len(data2[data2['Program Studi']==k])>40) (float(datapop[datapop['Program Studi']==k]['Populasi'])>50) &
        indexv = data2[ (data2['Program Studi'] == k) & (data2['Kapan anda memulai bekerja'] > uplimit) ].index
        # data2.drop(indexv[0:ceil(len(indexv)/3)] , inplace=True) # remove
        data2.loc[indexv,'Kapan anda memulai bekerja'] = data2[data2['Program Studi']==k]['Kapan anda memulai bekerja'].mean()#uplimit # replace
        uplimit = uplimit-1
        ind = ind+1
    jenjang = ['D3', 'D4', 'S1', 'S2']
    popu = [len(data[data['Program Studi'].str.startswith(k)]) for k in jenjang]
    popu = [sum(popu)] +popu
    respon = dict(zip(['Universitas', 'D3', 'D4', 'S1', 'S2'], popu))
    df_usaha=df1.copy()
    indx = df_usaha[df_usaha['Jenis perusahaan'].isna()].index
    df_usaha.drop(indx, inplace=True)

    fakultas = {"FIF": ['S1 Informatika', 'S1 Teknologi Informasi', 'S2 Informatika'],
                "FRI":['S1 Teknik Industri', 'S1 Sistem Informasi', 'S2 Teknik Industri'],
                "FTE":['S1 Teknik Telekomunikasi', 'S1 Teknik Elektro', 'S1 Teknik Komputer', 'S1 Teknik Fisika', 'S2 Teknik Elektro'],
                'FEB':['S1 International ICT Business', 'S2 Manajemen', 'S1 Akuntansi', 'S1 Manajemen'],
                'FKB':['S1 Ilmu Komunikasi (Int.)', 'S1 Administrasi Bisnis (Int.)', 'S1 Hubungan Masyarakat', 'S1 Ilmu Komunikasi', 'S1 Administrasi Bisnis'],
                'FIK':['S1 Seni Rupa', 'S1 Desain Produk', 'S1 Kriya', 'S1 Desain Interior', 'S1 Desain Komunikasi Visual'],
                'FIT':['D3 Sistem Informasi', 'D3 Teknologi Komputer', 'D3 Sistem Informasi Akuntansi', 'D3 Manajemen Pemasaran', 'D3 Teknologi Telekomunikasi', 'D3 Rekayasa Perangkat Lunak Aplikasi', 'D3 Perhotelan', 'D4 Teknologi Rekayasa Multimedia']}

    s=0
    for i,k in enumerate(fakultas):
        s=s+len(fakultas[k])

    data2 = data2.copy()
    data2['Berapa pendapatan yang anda peroleh']=data2['Berapa pendapatan yang anda peroleh'].astype('category')
    data2['Berapa pendapatan yang anda peroleh'].cat.reorder_categories(['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000'],  inplace=True)
    len(data2[data2['Berapa pendapatan yang anda peroleh'].isna()])
    data3=data2[data2['Berapa pendapatan yang anda peroleh'].notna()].copy()

    tag ='Tempat Bekerja'
    frek = df1[tag].value_counts()
    pdloc = pd.read_csv('data/loc-kerja.csv')
    tag ='Tempat Bekerja'
    frek = df1[tag].value_counts()
    loc = list(frek.keys())
    from geopy.geocoders import Nominatim
    # Initialize Nominatim API
    lons =[]
    lats =[]
    for i,k in enumerate(loc):
        geolocator = Nominatim(user_agent="iruma", timeout=100)
        location = geolocator.geocode(k)
        lons = lons+[location.longitude]
        lats =lats+[location.latitude]

    tag1='Posisi/jabatan anda saat ini (F5c)'
    tag2='Nama perusahaan tempat bekerja ? (F5b)'
    
    from wordcloud import WordCloud, STOPWORDS
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in df1[tag2]:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += ' '.join(tokens)+' '

    term =["Telkom indonesia|Telkom", "Telkom University|telkom university|Universitas Telkom|universitas telkom", "bank mandiri|mandiri|Mandiri", 'freelance|Freelance', 'astra|Astra', 'sigma|Sigma|media nusantara|Media Nusantara|Data Global|data global','huawei|Huawei', 'telkomsel|Telkomsel', 'bca|bank central|BCA', 'shopee|Shopee', 'pertamina|Pertamina']
    firm_ten=['PT. Telkom', 'Telkom University', 'Bank Mandiri', 'Freelance', 'PT. Astra', 'Telkom Sigma', 'Huawei', 'Telkomsel', 'BCA', 'Shopee', 'Pertamina']
    ls = [df1[tag2].str.count(k).sum() for k in term]
    firm_ten=[i for k,i in sorted(zip(ls,firm_ten))]
    ls.sort()

    from matplotlib.widgets import RectangleSelector
    col=['b', 'g', 'c', 'm', 'blueviolet', 'slateblue', 'lightcoral', 'brown']

    ############################################ DATA #########################################
    df_melanjutkan = df2.copy() # df2 adalah yang berstatus 'Melanjutkan Pendidikan'
    indx = df_melanjutkan[df_melanjutkan['Fakultas'].isna()].index
    df_melanjutkan.drop(indx, inplace=True)

    df_tidakbekerja = df3 # df3 adalah yang berstatus 'Tidak Bekerja'
    indx = df_tidakbekerja[df_tidakbekerja['Fakultas'].isna()].index
    df_tidakbekerja.drop(indx, inplace=True)

    df_wiraswasta = df4 # df4 adalah yang berstatus 'Wiraswasta'
    indx = df_wiraswasta[df_wiraswasta['Fakultas'].isna()].index
    df_wiraswasta.drop(indx, inplace=True)
    st.write('Jumlah Baris dengan Status Wiraswasta',len(df_wiraswasta))
    prodi_wiraswasta = list(df_wiraswasta['Program Studi'].unique())

    df_wiraswasta.rename(columns={
        df_wiraswasta.columns[77]: 'Sektor tempat anda berwirausaha',
        df_wiraswasta.columns[76]: 'Tingkat pendidikan apa yang paling tepat/sesuai dengan wirausaha anda?',
        df_wiraswasta.columns[75]: 'Jika wirausaha saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya?',
        df_wiraswasta.columns[58]: 'Posisi di wirausaha anda saat ini',
        df_wiraswasta.columns[78]: 'Berapa pendapatan yang anda peroleh dari wirausaha',
        df_wiraswasta.columns[66]: 'Kapan anda memulai berwirausaha'
        },inplace=True)
    
    # tidak ada lulusan dari prodi S1 Teknologi Informasi yang menjadi Wiraswasta
    prodi_wiraswasta = df_wiraswasta['Program Studi'].unique()

    df_wiraswasta.loc[df4['Kapan anda memulai berwirausaha'] < 0, 'Kapan anda memulai berwirausaha'] = 0
    df_wiraswasta = df_wiraswasta[df_wiraswasta['Kapan anda memulai berwirausaha'] <= 32]
    
    from math import ceil
    upbound = 30
    col = list(df_wiraswasta.columns)
    # del data2
    data4 = df_wiraswasta.copy() # df1 adalah yang berstatus 'Bekerja'
    data4["NIM"] = data4["NIM"].values.astype(str)
    prodi = list(df_wiraswasta['Program Studi'].unique())

    for i,k in enumerate(prodi):
        uplimit = upbound
        ind = 0
        idxv = data4[(data4['Kapan anda memulai berwirausaha'].isna()) & (data4['Program Studi']==k)].index
        idxv2 = data4[(data4['Kapan anda memulai berwirausaha'].notna()) & (data4['Program Studi']==k)].index
        rep = np.array(data4.loc[idxv2,'Kapan anda memulai berwirausaha'])
        while (data4[data4['Program Studi']==k]['Kapan anda memulai berwirausaha'].mean()>10):
            # print(data2[data2['Program Studi']==k]['Kapan anda memulai berwirausaha'].mean())
            indexv = data4[ (data4['Program Studi'] == k) & (data4['Kapan anda memulai berwirausaha'] > uplimit) ].index
            # data2.drop(indexv[0:ceil(len(indexv)/3)] , inplace=True) # remove
            data4.loc[indexv,'Kapan anda memulai berwirausaha'] = data4[data4['Program Studi']==k]['Kapan anda memulai berwirausaha'].mean()
            uplimit = uplimit-1
            ind = ind+1
    ###########################################################################################

    # UNIVERSITAS
    respon_rate(data, pop_dict)
    xn, param, wkt_tunggu = plot_overall(data2, data)
    gen_wkt_tunggu(data2, fakultas)
    tingkat_perusahaan_univ(df_usaha)
    sektor_perusahaan_univ(df_usaha)
    salary_an('FIF', data3, data2)
    gen_pendapatan(data2, fakultas)
    sebaran_univ(pdloc, df1)
    top_perusahaan_univ(firm_ten, ls)
    jabatan_univ(df1, tag1, tag2)
    status_pekerjaan_univ(data)

    frekuensi_melanjutkan_pendidikan(data, df_melanjutkan)
    sumber_biaya_melanjutkan_pendidikan(df_melanjutkan)
    sebaran_melanjutkan_pendidikan(df_melanjutkan, 'Nama Perguruan Tinggi')

    frekuensi_tidak_bekerja(data, df_tidakbekerja)
    level_motivasi_tidak_bekerja(df_tidakbekerja)

    frekuensi_wiraswasta(data, df_wiraswasta)
    sebaran_sektor_pekerjaan_wiraswasta(df_wiraswasta)
    sebaran_kesesuaian_pendidikan_wiraswasta(df_wiraswasta)
    sebaran_gaji_wiraswasta(df_wiraswasta)
    sebaran_posisi_jabatan_wiraswasta(df_wiraswasta)
    waktu_tunggu_wiraswasta(data4)
    # df_w=filtering_tempat_wiraswasta(df_wiraswasta)
    # pdloc=filtering_pdloc_wiraswasta(df_w)
    # tempat_wiraswasta(df_w, pdloc)
    status_perusahaan_wiraswasta(df_wiraswasta, 'Apa status usaha anda?', 'Apa tingkat tempat usaha/perusahaan anda')
    top_perusahaan_wiraswasta(df_wiraswasta)

    # FAKULTAS
    for i,k in enumerate(fakultas):
        fktunggu(k, data2, data, fakultas, pop_dict)
    for k in fakultas.keys():
        levelfirm(k, fakultas, df_usaha, col)
    for k in fakultas.keys():
        salaryrate(k, fakultas, data2, col)
    for k in fakultas.keys():
        fakultas_tempat(k, df1)
    tag='Status Anda saat ini (F8)'
    tags = ['Posisi/jabatan anda saat ini (F5c)', 'Nama perusahaan tempat bekerja ? (F5b)', 'Berapa pendapatan yang anda peroleh', 'Tempat Bekerja']
    for fak in fakultas.keys():
        plot_fakultas_jabatan(fak,12,(10,15), df1)

    for i,k in enumerate(fakultas):
        frekuensi_wiraswasta_fakultas(df_wiraswasta, k)
    for i,k in enumerate(fakultas):
        data=df_wiraswasta[df_wiraswasta['Fakultas']==k]['Sektor tempat anda berwirausaha'].value_counts()
        sebaran_sektor_pekerjaan_wiraswasta_fakultas(data, k, 12, (8,8))
    for i,k in enumerate(fakultas):
        sebaran_kesesuaian_pendidikan_wiraswasta_fakultas(df_wiraswasta, k)
    for i,k in enumerate(fakultas):
        sebaran_gaji_wiraswasta_fakultas(df_wiraswasta, k, 12)
    for i,k in enumerate(fakultas):
        sebaran_pendapatan_wiraswasta_fakultas(df_wiraswasta, k)
    for i,k in enumerate(fakultas):
        sebaran_posisi_jabatan_wiraswasta_fakultas(df_wiraswasta, k, 12)
    for i,k in enumerate(fakultas):
        waktu_tunggu_wiraswasta_fakultas(data4, k)
    # for i,k in enumerate(fakultas):
    #     x = filtering_pdloc_wiraswasta_fakultas(df_w, k)
    #     tempat_wiraswasta_fakultas(df_w, x)
    for i,k in enumerate(fakultas):
        status_perusahaan_wiraswasta_fakultas(df_wiraswasta, fakultas, k)
    for i,k in enumerate(fakultas):
        top_perusahaan_wiraswasta_fakultas(df_wiraswasta, k)
        
    # PRODI
    fns = 13
    for i, f in enumerate(fakultas):
        for k in fakultas[f]:
            prodi_tunggu(k, data2, data, pop_dict, fns)
    for i,f in enumerate(fakultas):
        for k in fakultas[f]:
            tingkat_prodi(k, df_usaha)
    tag = 'Sektor tempat anda bekerja'
    sektor = list(df_usaha[tag].unique())
    x = np.arange(len(sektor))
    width = 0.75
    frek = df_usaha[tag].value_counts()
    for lm in fakultas.keys():
        for i,k in enumerate(fakultas[lm]):
            dataprovin=df1[df1['Program Studi']==k][tag].value_counts()
            sektor_prodi(k,dataprovin, 12, (8,8), col[i])
    tag='Status Anda saat ini (F8)'
    tags = ['Posisi/jabatan anda saat ini (F5c)', 'Nama perusahaan tempat bekerja ? (F5b)', 'Berapa pendapatan yang anda peroleh', 'Tempat Bekerja']
    for fak in fakultas.keys():
        for k in fakultas[fak]:
            pendapatan_prodi(k,12,(10,15), df1)
    for k in fakultas.keys():
        for j in fakultas[k]:
            prodi2(j, df1)
    tag='Status Anda saat ini (F8)'
    tags = ['Posisi/jabatan anda saat ini (F5c)', 'Nama perusahaan tempat bekerja ? (F5b)', 'Berapa pendapatan yang anda peroleh', 'Tempat Bekerja']
    for fak in fakultas.keys():
        for k in fakultas[fak]:
            plot_prodi2(k,12,(10,15), df1)
    tag= ['Sektor tempat anda bekerja', 'Tingkat pendidikan apa yang paling tepat/sesuai dengan pekerjaan anda saat ini ? (F15)', 'Jika pekerjaan saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya ? (F16)']
    for lm in fakultas.keys():
        for i,k in enumerate(fakultas[lm]):
            distribusih(k, 14, (25,20), df1, pop_dict, data, col[i])
    
    for i,k in enumerate(prodi_wiraswasta):
        pop_dict_pordi = populasirespond[k]
        frekuensi_wiraswasta_prodi(df_wiraswasta, k, pop_dict_pordi)
    for i,k in enumerate(prodi_wiraswasta):
        data=df_wiraswasta[df_wiraswasta['Program Studi']==k]['Sektor tempat anda berwirausaha'].value_counts()
        sebaran_sektor_pekerjaan_wiraswasta_prodi(data, k, 12, (8,8))
    for i,k in enumerate(prodi_wiraswasta):
        sebaran_kesesuaian_pendidikan_wiraswasta_prodi(df_wiraswasta, k)
    for i,k in enumerate(prodi_wiraswasta):
        sebaran_gaji_wiraswasta_prodi(df_wiraswasta, k, 12)
    for i,k in enumerate(prodi_wiraswasta):
        sebaran_posisi_jabatan_wiraswasta_prodi(df_wiraswasta, k, 12)
    # for i,k in enumerate(prodi_wiraswasta):
    #     x = filtering_pdloc_wiraswasta_prodi(df_w, k)
    #     tempat_wiraswasta_prodi(df_w, x)
    for i,k in enumerate(prodi_wiraswasta):
        status_perusahaan_wiraswasta_prodi(df_wiraswasta, k)
    for i,k in enumerate(prodi_wiraswasta):
        top_perusahaan_wiraswasta_prodi(df_wiraswasta, k)

    plot_survey()

allFakultas = {"FIF": ['S1 Informatika', 'S1 Teknologi Informasi', 'S2 Informatika'],
            "FRI":['S1 Teknik Industri', 'S1 Sistem Informasi', 'S2 Teknik Industri'],
            "FTE":['S1 Teknik Telekomunikasi', 'S1 Teknik Elektro', 'S1 Teknik Komputer', 'S1 Teknik Fisika', 'S2 Teknik Elektro'],
            'FEB':['S1 International ICT Business', 'S2 Manajemen', 'S1 Akuntansi', 'S1 Manajemen'],
            'FKB':['S1 Ilmu Komunikasi (Int.)', 'S1 Administrasi Bisnis (Int.)', 'S1 Hubungan Masyarakat', 'S1 Ilmu Komunikasi', 'S1 Administrasi Bisnis'],
            'FIK':['S1 Seni Rupa', 'S1 Desain Produk', 'S1 Kriya', 'S1 Desain Interior', 'S1 Desain Komunikasi Visual'],
            'FIT':['D3 Sistem Informasi', 'D3 Teknologi Komputer', 'D3 Sistem Informasi Akuntansi', 'D3 Manajemen Pemasaran', 'D3 Teknologi Telekomunikasi', 'D3 Rekayasa Perangkat Lunak Aplikasi', 'D3 Perhotelan', 'D4 Teknologi Rekayasa Multimedia']}
listFakultas = allFakultas.keys()

listFak = ['fif','fri','fte','feb','fkb','fik','fit']

keterangan = ['Bekerja', 'Melanjutkan Pendidikan', 'Tidak Bekerja', 'Wiraswasta', 'Survey Pengguna']
keterangan2 = ['Bekerja', 'Wiraswasta', 'Survey Pengguna']
keterangan3 = ['Bekerja', 'Wiraswasta']


def logo():
     # Menambahkan gambar di sebelah judul
    logo_url = 'https://edurank.org/assets/img/uni-logos/telkom-university-logo.png'
    st.markdown(
        f'<img src="{logo_url}" alt="Logo Universitas" width="180" align="left">'
        '<h1 style="display: inline-block; margin-left: 20px;width: 400px;height:200px">Laporan Tracer Study Tahun 2022 Telkom University</h1>',
        unsafe_allow_html=True
    )

def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Universitas", 'Fakultas', 'Program Studi'])
        if selected == "Universitas":
            options_keterangan= st.selectbox(
                'Pilih Status:',
                keterangan) 
        elif selected == "Fakultas":
            options_faculty = st.selectbox(
                'Pilih Fakultas:',
                listFak, key="pilihFakultass")
            options_keterangan = st.selectbox(
                'Pilih Status:',
                keterangan2, key="pilihStatus"
            )
        elif selected == "Program Studi":
            options_faculty = st.selectbox(
                'Pilih Fakultas:',
                listFak, key="pilihFakultas")
            options_prodi= st.selectbox(
                'Pilih Prodi:',
                allFakultas[options_faculty], key="pilihProdi")
            if options_prodi in ['S1 Ilmu Komunikasi (Int.)', 'S1 Administrasi Bisnis (Int.)']:  
                options_keterangan = st.selectbox(
                    'Pilih Status:',
                    keterangan3, key="pilihStatuss"
                )
            else:
                options_keterangan = st.selectbox(
                    'Pilih Status:',
                    keterangan2, key="pilihStatuss"
                )
    
    # Jika salah satu item di option menu dipilih
    if selected == "Universitas":
        logo()
        universitas(options_keterangan) # Panggil fungsi universitas
    if selected == "Fakultas":
        st.header('Selamat datang !')
        fakultas(logofak, options_faculty, options_keterangan) # Panggil fungsi fakultas
    if selected == "Program Studi":
        st.header('Selamat datang !')
        programstudi(options_faculty, options_prodi, options_keterangan) # Panggil fungsi program studi

# Menampilkan/hasil
if __name__ == '__main__':
    main() 
