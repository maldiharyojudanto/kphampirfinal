import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from mpl_toolkits.basemap import Basemap, maskoceans
from gen_datvis import get_box_plot_data
from geopy.geocoders import Nominatim

import streamlit as st
from wordcloud import WordCloud, STOPWORDS
#WAKTU TUNGGU LULUSAN (PRODI)
def prodi_tunggu(k, data, dataori, pop_dict, fns):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(1)
    ax=plt.figure(figsize=[18, 16])
    dtfa =[np.array(data[data['Program Studi']==k]['Kapan anda memulai bekerja'])]
    temp = np.array([])
    for i in range(len(dtfa)):
        temp = np.append(temp,dtfa[i])

    dtfa = dtfa+[temp]
#     wkt = dict(zip(fakultas[fak]+['{} total'.format(fak)], dtfa))
    jumlah =[len(dataori[dataori['Program Studi']==k])]
    jumlah =jumlah+[sum(jumlah)]
    popu = [pop_dict[k]]
    popu = popu+[sum(popu)]
    
    axes = plt.subplot(1,1, 1) #facecolor='gainsboro'
    total = list(dtfa[0])+list(dtfa[1])
    n,bins,patches = plt.hist(total, color='#ff9999',width=1.5, bins=10,  edgecolor="black")
    plt.xlabel('bulan')
    xn =np.linspace(np.min(bins)-0.5,np.max(bins),100)
    expon = stats.gamma
    param = expon.fit(total)
    y = expon.pdf(xn, *param)*len(total)#interp1d(bins,np.histogram(wkt_tunggu,11)[0], kind='cubic')
    plt.plot(xn, y , linewidth=3, color='red')
    mode = axes.get_ylim()[1]*0.9
    plt.axvline(round(np.mean(dtfa[-1]),2), 0, 3000)
    plt.axvline(round(np.median(dtfa[-1]),2), 0, 3000, color='orange')
    plt.xticks(fontsize=fns)
    plt.yticks(fontsize=fns)
    plt.text(round(np.mean(dtfa[-1]),2), mode,'Mean= {}'.format(round(np.mean(dtfa[-1]),2)))
    plt.text(round(np.median(dtfa[-1]),2),mode/2,'Median= {}'.format(round(np.median(dtfa[-1]),2)))
    plt.savefig('fig/waktu_tunggu_{}.png'.format(k), dpi=300, transparent=True, bbox_inches='tight')

#TINGKAT PERUSAHAAN PRODI
def tingkat_prodi(k, df_usaha):
    bisnis = ['Nasional', 'Local', 'MultiNasional']
    x = np.arange(len(bisnis))
    width = 0.2
    frek = df_usaha[df_usaha['Program Studi']==k]['Jenis perusahaan'].value_counts()
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
    [plt.annotate('{}'.format(k), (i-0.1,k+200), fontsize=12) for i,k in enumerate(frek.values)]
    [plt.annotate('{}%'.format(round(k/sum(frek.values)*100,2)), (i-0.2,k/2), fontsize=12) for i,k in enumerate(frek.values)]
    plt.xticks(bisnis, rotation=20, fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig('fig/tingkat_perusahaan_{}.png'.format(k), dpi=300, transparent=True)

def sektor_prodi(var,dataprovin,fontsz, figsz, col='red'):
  # dataprovin = df1[a].value_counts()
  # dataprovin.unique()
  # dataprovin.value_counts()
  # plt.facecolor("yellow")
  fig, ax =plt.subplots(1,figsize=figsz, frameon=False)
  plt.xticks([],[])
  plt.yticks(rotation = 30, fontsize=fontsz)
  # portion = [[dataprovin[k]/pop_dict[k]*100,(pop_dict[k]-dataprovin[k])/pop_dict[k]*100] for i,k in enumerate(prodi)]
  ax.barh(dataprovin.keys(), [dataprovin[k] for k in dataprovin.keys()], color=col, edgecolor='black')
  # ax.barh(dataprovin.keys(), [ pop_dict[k]-dataprovin[k] for k in prodi], left=[dataprovin[k] for k in prodi], color="cyan", label='tidak merespon')
  _=[ax.annotate('{}% ({})'.format(round(k/sum(dataprovin.values)*100,2), k), (k+0.01,i), fontsize=10) for i,k in enumerate(dataprovin.values)]
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  ax.set_title('{}'.format(var))
  plt.tight_layout()
  plt.savefig('fig/sektor_perusahaan_{}.png'.format(var), dpi=100, transparent=True)
  # plt.show()

#PENDAPATAN PRODI
def pendapatan_prodi(var,fontsz, figsz, df1, col='red'):
  fig, ax = plt.subplots(1,figsize=figsz)
  tag = 'Berapa pendapatan yang anda peroleh'
  salary = df1[df1['Program Studi']==var][tag].value_counts()
  labels=['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000']
  ref = [4000000, 5500000, 8500000, 12500000, 20000000]
  midpoint = np.zeros(len(salary.keys()))
  for i,k in enumerate(salary.keys()):
    for j in range(len(labels)):
      if k==labels[j]:
        midpoint[i] =ref[j]
  ref=sorted(range(len(midpoint)), key=lambda k: midpoint[k])
  ax.set_title('{}'.format(var))
  ax.bar([salary.keys()[k] for k in ref], [salary.values[k] for k in ref], color="#ff9999", edgecolor='black')
  ax.tick_params(axis='x', labelrotation=30)
  mean_gaji = sum([salary.values[k]*midpoint[k]/sum(salary.values) for k in range(len(salary))])
  std_gaji = np.sqrt(sum([salary.values[k]*midpoint[k]**2/sum(salary.values) for k in range(len(salary))])-(mean_gaji**2))
  xmin, xmax, ymin, ymax = ax.axis()
  ax.annotate("Rata-rata gaji= Rp.{},-".format(math.floor(mean_gaji)),(xmin+0.5*(xmax-xmin),ymax-0.2*(ymax-ymin)), fontsize=12)
  # print(math.floor(mean_gaji))
  h=(ymax-ymin)/70
  ax.annotate("Standard deviasi gaji= Rp.{},-".format(math.floor(std_gaji)),(xmin+0.5*(xmax-xmin),ymax-0.3*(ymax-ymin)), fontsize=12)
  _ = [ax.annotate('{} %'.format(round(s/sum(salary.values)*100,1)),(k,s+h), fontsize=fontsz) for (k,s) in zip([salary.keys()[j] for j in ref],[salary.values[k] for k in ref])]
  plt.savefig('fig/pendapatan_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')

#Sebaran Lokasi Kerja Prodi
'''def prodi2(var, df1):
  tag ='Tempat Bekerja'
  frek = df1[df1['Program Studi']==var][tag].value_counts()
  loc = list(frek.keys())
  # Import the required library
  # Initialize Nominatim API
  lons =[]
  lats =[]
  for i,k in enumerate(loc):
    geolocator = Nominatim(user_agent="iruma", timeout=100)
    location = geolocator.geocode(k)
    lons = lons+[location.longitude]
    lats =lats+[location.latitude]
    # print(i,k)
  locdict= {'name':frek.keys(),'pop':frek.values, 'lat':lats, 'lon':lons}
  pdloc=pd.DataFrame(locdict)
  lat1= -10.847
  lat2= 12
  lon1=94
  lon2= 143.042
  lon=np.linspace(lon1,lon2,100)
  lat=np.linspace(lat1,lat2,60)
  print(lon[1]-lon[0], lat[1]-lat[0])
  lons,lats = np.meshgrid(lon,lat)

  fig,ax=plt.subplots(1,1,figsize=(30,30)) #
  m = Basemap(projection='merc',llcrnrlat=round(lat1,3),urcrnrlat=round(lat2,3),llcrnrlon=round(lon1,3),urcrnrlon=round(lon2,3),resolution='l', ax=ax)
  x,y = m(pdloc['lon'],pdloc['lat'])
  # m.drawparallels(np.arange(lat1,lat2,6),labels=[1,0,0,0],fontsize=10)
  # m.drawmeridians(np.arange(lon1,lon2,12),labels=[0,0,0,1],fontsize=10)
  m.drawcoastlines()
  m.fillcontinents(color='aquamarine')
  # prepare a color for each point depending on the continent.
  pdloc['labels_enc'] = pd.factorize(pdloc['name'])[0]

  # Add a point per position
  m.scatter(
      x,
      y,
      s=(pdloc['pop']-pdloc['pop'].min())/(pdloc['pop'].max()-pdloc['pop'].min())*10000,
      alpha=0.7,
      c=pdloc['labels_enc'],
      cmap="Set1",
      zorder=5
  )
  # plt.show()
  ax.set_title('{}'.format(var))
  ax2 = fig.add_axes([0.705, 0.51, 0.18, 0.18])
  province = df1[df1['Program Studi']==var]['Provinsi tempat bekerja'].value_counts()
  province=province[0:10]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  # plt.xlim(0,3000)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.savefig(fname='fig/sebaran_lokasi_kerja_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')'''

def prodi2(var, df1):
  tag ='Tempat Bekerja'
  frek = df1[df1['Program Studi']==var][tag].value_counts()
  loc = list(frek.keys())
  # Import the required library
  # Initialize Nominatim API
  lons =[]
  lats =[]
  for i,k in enumerate(loc):
    geolocator = Nominatim(user_agent="iruma", timeout=100)
    location = geolocator.geocode(k)
    lons = lons+[location.longitude]
    lats =lats+[location.latitude]
    # print(i,k)
  locdict= {'name':frek.keys(),'pop':frek.values, 'lat':lats, 'lon':lons}
  pdloc=pd.DataFrame(locdict)
  lat1= -10.847
  lat2= 12
  lon1=94
  lon2= 143.042
  lon=np.linspace(lon1,lon2,100)
  lat=np.linspace(lat1,lat2,60)
  print(lon[1]-lon[0], lat[1]-lat[0])
  lons,lats = np.meshgrid(lon,lat)
  print(var)
  fig,ax=plt.subplots(1,1,figsize=(30,30)) #
  m = Basemap(projection='merc',llcrnrlat=round(lat1,3),urcrnrlat=round(lat2,3),llcrnrlon=round(lon1,3),urcrnrlon=round(lon2,3),resolution='l', ax=ax)
  x,y = m(pdloc['lon'],pdloc['lat'])
  # m.drawparallels(np.arange(lat1,lat2,6),labels=[1,0,0,0],fontsize=10)
  # m.drawmeridians(np.arange(lon1,lon2,12),labels=[0,0,0,1],fontsize=10)
  m.drawcoastlines()
  m.fillcontinents(color='aquamarine')
  # prepare a color for each point depending on the continent.
  pdloc['labels_enc'] = pd.factorize(pdloc['name'])[0]

  # Add a point per position
  m.scatter(
      x,
      y,
      s=(pdloc['pop']-pdloc['pop'].min())/(pdloc['pop'].max()-pdloc['pop'].min()+0.0001)*10000,
      alpha=0.7,
      c=pdloc['labels_enc'],
      cmap="Set1",
      zorder=5
  )
  # plt.show()
  ax.set_title('{}'.format(var))
  ax2 = fig.add_axes([0.705, 0.51, 0.18, 0.18])
  province = df1[df1['Program Studi']==var]['Provinsi tempat bekerja'].value_counts()
  province=province[0:10]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  # plt.xlim(0,3000)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.savefig(fname='fig/sebaran_lokasi_kerja_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')
  # plt.show()

#Jabatan/Posisi Kerja Prodi
def plot_prodi2(var,fontsz, figsz, df1, col='red'):
  fig, ax = plt.subplots(1,figsize=figsz)
  tag='Posisi/jabatan anda saat ini (F5c)'
  posisi=df1[df1['Program Studi']==var][tag].value_counts()
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  ax.set_yticklabels(posisi.keys(),fontsize=12)
  ax.set_title('{}'.format(var))
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100,1)),(k,i), fontsize=fontsz) for i,k in enumerate(posisi.values)]
  plt.savefig('fig/jabatan_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')

#Responrate Prodi  
# Perprodi: sektor,tempat, kesesuaian level, alasan jika tidak sesuai  
def distribusih(var,fontsz, figsz, df1, pop_dict, data, col='red'):
  tag='Sektor tempat anda bekerja'
  dataprovin = df1[df1['Program Studi']==var][tag].value_counts()
  dataprovin.value_counts()
  fig, ax =plt.subplots(1,1,figsize=figsz)
  plt.suptitle('{}'.format(var))
  # portion = [[dataprovin[k]/pop_dict[k]*100,(pop_dict[k]-dataprovin[k])/pop_dict[k]*100] for i,k in enumerate(prodi)]
  labl=['Merespon', 'Tidak merespon']
  dat = data[data['Program Studi']==var].copy()
  val = [len(dat), pop_dict[var]-len(dat)]
  pval = [round(v/sum(val)*100,1) for v in val]
  explode=(0, 0.05)
  print(pval)
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  # Set aspect ratio to be equal so that pie is drawn as a circle.
  # plt.axis('equal')
  plt.savefig('fig/respon_rate_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')

################################################ WIRASWASTA ############################################
def frekuensi_wiraswasta_prodi(df_wiraswasta, varprodi, pop_dict): # LEVEL PRODI
  frek_wiraswasta_prodi = df_wiraswasta['Program Studi'].value_counts()[varprodi]
  
  labl = [varprodi+' (Berwirausaha)', varprodi+' (Semua Mahasiswa)']
  val = [frek_wiraswasta_prodi, pop_dict]
  # print(val)
  pval = [round(v/sum(val)*100,1) for v in val]
  explode = (0, 0.05) # jarak pada chart
  # print(pval)
  fig, ax =plt.subplots(figsize=(8,6)) # 1: respon 2: sektor 3:tingkat pend 4:jika tidak sesuai
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  plt.annotate('Populasi (Semua Responden) {} = {}'.format(varprodi, pop_dict), (-1.2,0.8))
  plt.tight_layout()
  plt.savefig('fig/frekuensi-wiraswasta-prodi-{}.png'.format(varprodi), dpi=300, transparent=True)

def sebaran_sektor_pekerjaan_wiraswasta_prodi(df_wiraswasta, var, fontsz, figsz): # LEVEL PRODI
  fig, ax =plt.subplots(1,figsize=figsz, frameon=False)
  # plt.title(var, size=16)
  plt.xticks([],[])
  plt.yticks(rotation = 30, fontsize=fontsz)
  ax.barh(df_wiraswasta.keys(), [df_wiraswasta[k] for k in df_wiraswasta.keys()], color='lightcoral', edgecolor='black')
  _=[ax.annotate('{}% ({})'.format(round(k/sum(df_wiraswasta.values)*100,2), k), (k+0.01,i), fontsize=10) for i,k in enumerate(df_wiraswasta.values)]
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/sebaran-sektor-pekerjaan-wiraswasta-prodi-{}.png'.format(var), dpi=100, transparent=True)

def sebaran_kesesuaian_pendidikan_wiraswasta_prodi(df_wiraswasta, var): # LEVEL PRODI
  tag='Tingkat pendidikan apa yang paling tepat/sesuai dengan pekerjaan anda saat ini ? (F15).1'
  datavar = df_wiraswasta[df_wiraswasta['Program Studi']==var][tag].value_counts()
  # order = [1,0,2,3]
  fig, ax=plt.subplots(figsize=(20,10))
  ax.barh(datavar.keys(), datavar.values, color='slateblue', edgecolor='black')
  plt.yticks(rotation = 40, fontsize=13)
  plt.xticks(rotation = 40, fontsize=13)
  _=[ax.annotate('  {}%'.format(round(datavar.values[k]/sum(datavar.values)*100)), (datavar.values[k],k), fontsize=14) for k in range(len(datavar))]
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-prodi-{}.png'.format(var), dpi=100, transparent=True)
  
  tag2 = 'Jika pekerjaan saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya ? (F16).1'
  frek_pertanyan = df_wiraswasta[df_wiraswasta['Program Studi']==var][tag2].value_counts()
  ques = ['Q{}'.format(k) for k in range(1,len(frek_pertanyan))]
  ques2 = ['Q1: Prospek karir lebih baik', 'Q2: Belum mendapat yang lebih sesuai', 'Q3:Lebih menarik', 'Q4:Lebih menyukai karir saat ini', 'Q5:Pengalaman kerja', 'Q6: Lebih terjamin', 'Q7: Lokasi lebih dekat rumah', 'Q8:Pendapatan tinggi','Q9:Waktu lebih fleksibel', 'Q10:Dipromosikan', 'Q11:Lebih menjamin kebutuhan keluarga.']
  fig, ax=plt.subplots(figsize=(20,10))
  ax.bar(np.arange(len(ques)), frek_pertanyan.values[1:], edgecolor='black', color='blue')
  xmin, xmax, ymin, ymax = ax.axis()
  _ = [ax.annotate(k, ((xmax+xmin)/4,ymax-(i+1)*(ymax-ymin)/20)) for i,(k,c) in enumerate(zip(ques2,frek_pertanyan.values[1:]))]
  plt.xticks(np.arange(len(ques)),ques)
  plt.tight_layout()
  plt.savefig('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-prodi-{}.png'.format(var), dpi=100, transparent=True)

def sebaran_gaji_wiraswasta_prodi(df_wiraswasta, var, fontsz): # LEVEL PRODI
  tag = 'Berapa pendapatan yang anda peroleh.1'
  salary = df_wiraswasta[df_wiraswasta['Program Studi']==var][tag].value_counts()
  labels=['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000']
  ref = [4000000, 5500000, 8500000, 12500000, 20000000]
  midpoint = np.zeros(len(salary.keys()))
  for i,k in enumerate(salary.keys()):
    for j in range(len(labels)):
      if k==labels[j]:
        midpoint[i] =ref[j]
  ref=sorted(range(len(midpoint)), key=lambda k: midpoint[k])
  fig, ax = plt.subplots(figsize=(10,8))
  ax.bar([salary.keys()[k] for k in ref], [salary.values[k] for k in ref], color="#ff9999", edgecolor='black')
  ax.tick_params(axis='x', labelrotation=30)
  mean_gaji = sum([salary.values[k]*midpoint[k]/sum(salary.values) for k in range(len(salary))])
  std_gaji = np.sqrt(sum([salary.values[k]*midpoint[k]**2/sum(salary.values) for k in range(len(salary))])-(mean_gaji**2))
  xmin, xmax, ymin, ymax = ax.axis()
  ax.annotate("Rata-rata gaji wiraswasta= Rp.{},-".format(math.floor(mean_gaji)),(xmin+0.5*(xmax-xmin),ymax-0.2*(ymax-ymin)), fontsize=12)
  # print(math.floor(mean_gaji))
  h=(ymax-ymin)/70
  ax.annotate("Standard deviasi gaji wiraswasta= Rp.{},-".format(math.floor(std_gaji)),(xmin+0.5*(xmax-xmin),ymax-0.3*(ymax-ymin)), fontsize=12)
  _ = [ax.annotate('{} %'.format(round(s/sum(salary.values)*100,1)),(k,s+h), fontsize=fontsz) for (k,s) in zip([salary.keys()[j] for j in ref],[salary.values[k] for k in ref])]
  plt.tight_layout()
  plt.savefig('fig/sebaran-gaji-wiraswasta-prodi-{}.png'.format(var), dpi=300, transparent=True)

def sebaran_posisi_jabatan_wiraswasta_prodi(df_wiraswasta, var, fontsz): # LEVEL FAKULTAS
  fig, ax = plt.subplots(figsize=(8,6))
  tag='Posisi/jabatan anda saat ini (F5c).1'
  posisi=df_wiraswasta[df_wiraswasta['Program Studi']==var][tag].value_counts()
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  ax.set_yticklabels(posisi.keys(),fontsize=12)
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100,1)),(k,i), fontsize=fontsz) for i,k in enumerate(posisi.values)]
  plt.tight_layout()
  plt.savefig('fig/sebaran-posisi-jabatan-wiraswasta-prodi-{}.png'.format(var), dpi=100, transparent=True)

def filtering_pdloc_wiraswasta_prodi(df_wiraswasta, var):
  tag ='Di Kota/Kabupaten mana lokasi tempat usaha Anda?'
  frek = df_wiraswasta[(df_wiraswasta['Daerah Valid'] == 1) & (df_wiraswasta['Program Studi'] == var)][tag].value_counts()

  loc = list(frek.keys())
  print(len(frek))

  # Import the required library
  # Initialize Nominatim API
  lons =[]
  lats =[]
  namakota = []
  for i,k in enumerate(loc):
    geolocator = Nominatim(user_agent="iruma", timeout=100)
    location = geolocator.geocode(k)
    if location is not None and location.longitude is not None:
      namakota = namakota+[k]
      # print(loc[i])
      # print(location.longitude)
      lons = lons+[location.longitude]
      lats =lats+[location.latitude]
    # print(i,k)

  locdict= {'name':namakota, 'pop':frek.values ,'lat':lats, 'lon':lons}
  # print(len(locdict))
  pdloc=pd.DataFrame(locdict)
  # print(len(pdloc))
  # print(namakota)
  # locdict= {'name':frek.keys(),'pop':frek.values, 'lat':lats, 'lon':lons}
  # pdloc=pd.DataFrame(locdict)
  return pdloc

def tempat_wiraswasta_prodi(df_wiraswasta,dloc,var): #LEVEL PRODI
  tag ='Di Kota/Kabupaten mana lokasi tempat usaha Anda?'
  lat1= -10.847
  lat2= 12
  lon1=94
  lon2= 143.042
  lon=np.linspace(lon1,lon2,100)
  lat=np.linspace(lat1,lat2,60)
  print(lon[1]-lon[0], lat[1]-lat[0])
  lons,lats = np.meshgrid(lon,lat)

  fig,ax=plt.subplots(1,1,figsize=(30,30)) #
  m = Basemap(projection='merc',llcrnrlat=round(lat1,3),urcrnrlat=round(lat2,3),llcrnrlon=round(lon1,3),urcrnrlon=round(lon2,3),resolution='l', ax=ax)
  x,y = m(dloc['lon'],dloc['lat'])
  # m.drawparallels(np.arange(lat1,lat2,6),labels=[1,0,0,0],fontsize=10)
  # m.drawmeridians(np.arange(lon1,lon2,12),labels=[0,0,0,1],fontsize=10)
  m.drawcoastlines()
  m.fillcontinents(color='aquamarine')
  # plt.scatter(x,y)
  # prepare a color for each point depending on the continent.
  dloc['labels_enc'] = pd.factorize(dloc['name'])[0]

  if len(dloc['pop']) == 1:
    bubble = dloc['pop']*1000
    alp = 0.9
  else:
    bubble = (dloc['pop']-dloc['pop'].min())/(dloc['pop'].max()-dloc['pop'].min())*10000
    alp = 0.7
  # Add a point per position
  m.scatter(
      x,
      y,
      s=bubble,
      alpha=alp,
      c=dloc['labels_enc'],
      cmap="Set1",
      zorder=5
  )
  # plt.show()
  ax2 = fig.add_axes([0.705, 0.51, 0.18, 0.18])
  province = df_wiraswasta[(df_wiraswasta['Daerah Valid'] == 1) & (df_wiraswasta['Program Studi'] == var)][tag].value_counts()
  province=province[province.values>=20]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  plt.xlim(0,400)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.savefig('fig/sebaran-lokasi-perusahaan-wiraswasta-prodi-{}.png'.format(var), bbox_inches='tight', dpi=300, transparent=True)

def status_perusahaan_wiraswasta_prodi(df_wiraswasta, var): # LEVEL FAKULTAS
  prodi = list(df_wiraswasta[df_wiraswasta['Program Studi']==var]['Apa status usaha anda?'].unique())
  frek = df_wiraswasta[df_wiraswasta['Program Studi']==var]['Apa status usaha anda?'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  for i,k in enumerate(frek.values):
    plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
    plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.1,k/2), fontsize=12)
  plt.xticks(prodi, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,frek[0]+10)
  plt.tight_layout()
  # plt.savefig('fig/frekuensi-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)
  plt.savefig('fig/perusahaan-hukum-nonhukum-wiraswasta-prodi-{}.png'.format(var), dpi=300, transparent=True)

  prodi = list(df_wiraswasta[df_wiraswasta['Program Studi']==var]['Apa tingkat tempat usaha/perusahaan anda'].unique())
  frek = df_wiraswasta[df_wiraswasta['Program Studi']==var]['Apa tingkat tempat usaha/perusahaan anda'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  for i,k in enumerate(frek.values):
    plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
    plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.1,k/2), fontsize=12)
  plt.xticks(prodi, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,frek[0]+10)
  plt.tight_layout()
  # plt.savefig('fig/frekuensi-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)
  plt.savefig('fig/perusahaan-nasional-multinasional-wiraswasta-prodi-{}.png'.format(var), dpi=300, transparent=True)

def top_perusahaan_wiraswasta_prodi(df_wiraswasta, var): # PROGRAM STUDI
  tag1='Posisi/jabatan anda saat ini (F5c).1'
  tag2='Nama Perusahaan/Usaha Anda ? (F5b)'
  comment_words = ''
  stopwords = set(STOPWORDS)
  df = df_wiraswasta[df_wiraswasta['Program Studi']==var][tag2]

  # iterate through the csv file
  for val in df:

      # typecaste each val to string
      val = str(val)

      # split the value
      tokens = val.split()

      # Converts each token into lowercase
      for i in range(len(tokens)):
          tokens[i] = tokens[i].lower()

      comment_words += ' '.join(tokens)+' '

  wordcloud = WordCloud(width = 800, height = 800,
                  background_color ='white',
                  stopwords = stopwords,
                  min_font_size = 10).generate(comment_words)

  plt.figure(figsize = (8, 8), facecolor = None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  st.set_option('deprecation.showPyplotGlobalUse', False) # untuk menghilangkan warning bawaan dari streamlit
  plt.tight_layout()
  plt.savefig('fig/top-perusahaan-wiraswasta-prodi-{}.png'.format(var), dpi=300, transparent=True)

  # print(type(text))
  # print(text)
  wordcloud.words_.keys()
  term_wirausaha =["PT|pt|CV|cv|Indonesia|indonesia|id|putra|mandiri", "Toko|toko|shop|store|Coffe|coffe", 'Freelance|freelance|lepas', 'Desain|desain|desain interior|studio|seniman|interior','Tour|tour|Travel|travel|Laundry|laundry']
  firm_ten_wirausaha=['Pengusaha', 'Pedagang', 'Pekerja Lepas', 'Desainer', 'Bidang Jasa']
  ls = [df.str.count(k).sum() for k in term_wirausaha]
  # print(len(ls),ls)
  # print(ls)
  firm_ten_wirausaha=[i for k,i in sorted(zip(ls,firm_ten_wirausaha))]
  ls.sort()
  # print(firm_ten_wirausaha, ls)
  
  fig, ax=plt.subplots(figsize=(8,6))
  ax.barh(firm_ten_wirausaha,ls, color='salmon', edgecolor='black')
  ax.set_yticklabels(firm_ten_wirausaha,fontsize=12)
  for i,k in enumerate(ls):
    if k == 0:
      return
    else:
      ax.annotate('{}%'.format(round(k/sum(ls)*100)),(k,i), fontsize=12)
  plt.tight_layout()
  plt.savefig('fig/10-perusahaan-wiraswasta-prodi-{}.png'.format(var), dpi=300, transparent=True)