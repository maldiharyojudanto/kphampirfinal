import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from mpl_toolkits.basemap import Basemap, maskoceans
from gen_datvis import get_box_plot_data
from geopy.geocoders import Nominatim
from matplotlib.widgets import RectangleSelector

import streamlit as st
from wordcloud import WordCloud, STOPWORDS
#WAKTU TUNGGU LULUSAN & Responrate Fakultas
fns=13
def fktunggu(fak, data, dataori, fakultas, pop_dict):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(1)
    ax=plt.figure(figsize=[18, 16])
    # plt.suptitle('Waktu tunggu {}'.format(fak))
    dtfa =[np.array(data[data['Program Studi']==k]['Kapan anda memulai bekerja']) for k in fakultas[fak]]
    temp = np.array([])
    for i in range(len(dtfa)):
        temp = np.append(temp,dtfa[i])

    dtfa = dtfa+[temp]
    wkt = dict(zip(fakultas[fak]+['{} total'.format(fak)], dtfa))
    jumlah =[len(dataori[dataori['Program Studi']==k]) for k in fakultas[fak]]
    jumlah =jumlah+[sum(jumlah)]
    popu = [pop_dict[k] for k in fakultas[fak]]
    popu = popu+[sum(popu)]
    
    axes = plt.subplot(1,1, 1) #facecolor='gainsboro'
    total = list(dtfa[0])+list(dtfa[1])+list(dtfa[2])
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
    plt.savefig('fig/waktu_tunggu_{}.png'.format(fak), dpi=300, bbox_inches='tight')
    
    ax=plt.subplot(2, 2, 1) #facecolor='gainsboro'
    bp=plt.boxplot(dtfa, vert=False, showfliers=True, labels=wkt.keys(), widths=0.8, showmeans=True, meanline=True)
    plt.xlim(0,max([np.max(k) for k in dtfa])+20)
    templen=np.array([len(wkt[k]) for k in wkt.keys()])
    templen[np.isnan(templen)] = 0
    # [plt.text(17,i+1.1,'{}'.format(templen[i])) for i in range(len(wkt.keys()))]
    plt.xticks(fontsize=fns)
    templen=np.array([round(np.mean(wkt[k]),1) for k in wkt.keys()])
    templen[np.isnan(templen)] = 0
    # [plt.text(3,i+1.2,'{}'.format(templen[i])) for i in range(len(wkt.keys()))]
    plt.yticks(fontsize=fns)
    # plt.title('Distribusi waktu tunggu fakultas {}'.format(fak), fontsize=10)
    ax.annotate('Jumlah resp.', (data['Kapan anda memulai bekerja'].max()+4,len(jumlah)+0.3))
    dbp = get_box_plot_data(list(wkt.keys()), bp)
    _ = [ax.annotate('{}'.format(jumlah[i]), (data['Kapan anda memulai bekerja'].max()+4, i+0.9), fontsize=fns) for i,k in enumerate(list(wkt.keys()))]
    _ = [ax.annotate('{}'.format(round(np.mean(wkt[k]),2)), (np.mean(wkt[k]), i+1.2)) for i,k in enumerate(list(wkt.keys()))]
    _ = [ax.annotate('{}'.format(round(np.median(wkt[k]),1)), (np.median(wkt[k]), i+0.8)) for i,k in enumerate(list(wkt.keys()))]
    # _ = [ax.annotate('{}; {}% pencilan'.format(round(dbp.loc[i,'upper_whisker'],1),
    #                                                 round(len([elem for elem in list(wkt.values())[i] if elem > dbp.loc[i,'upper_whisker']])/len(list(wkt.values())[i])*100),1),
    #                       (dbp.loc[i,'upper_whisker']-0.5, i+1.3), fontsize=11) for i in range(len(dbp))]

    plt.subplot(2, 2, 2)
    pie =plt.pie([len(wkt[k]) for k in list(wkt.keys())[0:-1]], autopct='%1.0f%%',
          shadow=False, startangle=90, pctdistance=1.1, labeldistance=1.3, textprops={'fontsize': 11})
    # plt.legend(pie, labels=wkt.keys(), loc='upper', bbox_to_anchor=(-0.2, 1.2),
    #          fontsize=8)
    plt.legend(pie, labels=list(wkt.keys())[0:-1],bbox_to_anchor=(0, 1, 1, 0), loc='lower left', ncol=2, fontsize=fns) #, mode="expand"
    plt.title('Proporsi responder fakultas {}'.format(fak), fontsize=10, y=-0.01)
    # plt.setp(pie[1], rotation_mode="anchor", ha="center", va="center")
    # for i,tx in enumerate(pie[1]):
    #   if i !=2:
    #     rot = tx.get_rotation()
    #     tx.set_rotation(rot-90+(1-rot//180)*180)

    # stats.mode()[1]
    # plt.xticks(np.arange(0, 16, 1.0))
    axe = plt.subplot(2,2,3) # , facecolor='gainsboro'
    cross_tab = pd.DataFrame({'index':fakultas[fak]+['Total'], 'merespon':jumlah,'tidak merespon':[popu[i]-jumlah[i] for i,k in enumerate(fakultas[fak]+['Total'])]}).set_index('index')
    # axadd = fig.add_axes([0.3, 0.06, 0.2, 0.1])
    cross_tab.plot(kind='barh', stacked=True, colormap='tab10', ax=axe)
    plt.legend(loc=7)
    lncr=len(cross_tab)
    _ = [axe.annotate('{} % (Pop. = {})'.format(round(jumlah[i]/popu[i]*100,2),int(popu[i])), (popu[i], i), fontsize=10) for i in range(len(fakultas[fak]+['Total']))]
    axe.set_xlim(0,popu[-1]+2000)
    axe.set_xticks([], [])
    axe.set_ylabel(None)
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.spines['bottom'].set_visible(False)
    axe.spines['left'].set_visible(False)
    ## single boxplot FIF
    # plt.boxplot(total, vert=False,widths=0.8, showmeans=True, meanline=True )
    # [plt.text(2,1.1,'mean={}'.format(round(np.mean(total))))]
    # plt.xticks(fontsize=8)
    # quantiles = np.quantile(total, np.array([0.00, 0.25, 0.50, 0.75, 1.00]))
    # plt.vlines(quantiles, [0] * quantiles.size, [1] * quantiles.size,
    #           color='b', ls=':', lw=0.5, zorder=10)
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = True, bottom = True)
    # [plt.annotate('Q{}'.format(i), (quantiles[i], 0)) for i in range(1,4)]
    # plt.xticks(quantiles, fontsize=10)
    # plt.xticks(np.arange(-1, 16, 1.0))
    # plt.figtext(0.5,0.5, "{}".format(list(fakultas.keys())[0]))
    plt.tight_layout()
    #print("Mean S2 TI {}, S1 SI {}, S1 TI {}, Total {} ".format(np.mean(dtfa[0]),np.mean(dtfa[1]),np.mean(dtfa[2]),np.mean(total)))
    #print("Q2 S2 IF {}, S1 IF {}, S1 IT {}, Total {} ".format(np.percentile(dtfa[0], 50),np.percentile(dtfa[1], 50),np.percentile(dtfa[2], 50),np.percentile(total, 50)))
    plt.savefig('fig/respon_rate_{}.png'.format(fak), dpi=300, transparent=True)

#TINGKAT PERUSAHAAN Fakultas

def levelfirm(fak, fakultas, df_usaha, col):
  bisnis = list(df_usaha['Jenis perusahaan'].unique())
  bisnis[-1]='Wiraswasta'
  width = 0.9
  frek = []
  for k in fakultas[fak]:
    frek = frek +[df_usaha[df_usaha['Program Studi']==k]['Jenis perusahaan'].value_counts()]
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  n=len(frek)
  # print(x)
  rects = [ax.bar((np.arange(len(frek[k]))-width/n*(n-1))+k*width/n,frek[k].values, width/(n+0.2), label=fakultas[fak][k], color=col[k]) for k in range(n)]
  # [ax.annotate() for k in frek[n]]
  for i,k in enumerate(fakultas[fak]):
    x = np.arange(len(frek[i]))-width/n*(n-1)
    [ax.annotate('{} %'.format(round(p/sum(frek[i].values)*100)), (x[m]+(i-0.2)*width/n,p+0.2), fontsize=11) for m,p in enumerate(frek[i].values)]

  plt.xticks(np.arange(len(bisnis))-0.5, bisnis)
  ax.legend()
  # _=[ax.bar_label(k, padding=3) for k in rects]
  # ax.bar_label(rects2, padding=3)
  print(frek[0], type(frek[0]))
  plt.savefig('fig/tingkat_perusahaan_{}.png'.format(fak), bbox_inches='tight', dpi=300, transparent=True)

#PENDAPATAN Fakultas
def salaryrate(fak, fakultas, data2, col):
  bisnis = list(data2['Berapa pendapatan yang anda peroleh'].unique())
  width = 0.9
  frek = []
  labels=['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000']
  lab2=['< 4 juta', '4 juta - 7 juta', '7 juta - 10 juta', '10 juta - 15 juta', '>15 juta', '']
  n=len(fakultas[fak])
  fig, ax = plt.subplots(1,1,figsize=(12,8))
  for o,k in enumerate(fakultas[fak]):
    frek = frek +[data2[data2['Program Studi']==k ]['Berapa pendapatan yang anda peroleh'].value_counts()]
    ref = [4000000, 5500000, 8500000, 12500000, 20000000]
    midpoint = np.zeros(len(frek[o].keys()))
    for i,s in enumerate(frek[o].keys()):
      for j in range(len(labels)):
        if s==labels[j]:
          midpoint[i] =ref[j]
    ref=sorted(range(len(midpoint)), key=lambda k: midpoint[k])
    print([frek[o].keys()[l] for l in ref])
    ax.bar((np.arange(len(frek[o]))-width/n*(n-1))+o*width/n,[frek[o].values[k] for k in ref], width/(n+0.2), label=fakultas[fak][o], color=col[o])
    x = np.arange(len(frek[o]))-width/n*(n-1)
    [ax.annotate('{} %'.format(round(p/sum(frek[o].values)*100)), (x[m]+(o-0.3)*width/n,p+1), fontsize=12, color=col[o]) for m,p in enumerate([frek[o].values[l] for l in ref])]
    xmin,xmax, ymin, ymax = ax.axis()
    
  plt.xticks(np.arange(len(bisnis))-0.5, lab2)
  ax.tick_params(axis='x', labelrotation=30)
  ax.legend()
  plt.savefig('fig/pendapatan_{}.png'.format(fak), dpi=300, transparent=True, bbox_inches='tight')

#Sebaran Lokasi Kerja Fakultas
def fakultas_tempat(var, df1):
  tag ='Tempat Bekerja'
  frek = df1[df1['Fakultas']==var][tag].value_counts()
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
  province = df1[df1['Fakultas']==var]['Provinsi tempat bekerja'].value_counts()
  province=province[0:10]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  # plt.xlim(0,3000)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.savefig('fig/sebaran_lokasi_kerja_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')
  # plt.show()

#Jabatan/Posisi Kerja Fakultas
def plot_fakultas_jabatan(var,fontsz, figsz, df1, col='red'):
  fig, ax = plt.subplots(1,figsize=figsz)
  tag='Posisi/jabatan anda saat ini (F5c)'
  posisi=df1[df1['Fakultas']==var][tag].value_counts()
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  ax.set_yticklabels(posisi.keys(),fontsize=12)
  ax.set_title('{}'.format(var))
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100,1)),(k,i), fontsize=fontsz) for i,k in enumerate(posisi.values)]
  plt.savefig('fig/jabatan_{}.png'.format(var), dpi=300, transparent=True, bbox_inches='tight')

################################################ WIRASWASTA ############################################
def frekuensi_wiraswasta_fakultas(df_wiraswasta, var): # LEVEL FAKULTAS
  frek_wiraswasta_prodi = df_wiraswasta['Fakultas'].value_counts()[var]
  
  labl = [var+' (Berwirausaha)', 'Wiraswasta Universitas']
  val = [frek_wiraswasta_prodi, len(df_wiraswasta)]
  # print(val)
  pval = [round(v/sum(val)*100,1) for v in val]
  explode = (0, 0.05) # jarak pada chart
  # print(pval)
  fig, ax =plt.subplots(figsize=(8,6)) # 1: respon 2: sektor 3:tingkat pend 4:jika tidak sesuai
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  plt.annotate('Populasi (Semua Responden) {} = {}'.format(var, len(df_wiraswasta)), (-1.2,0.8))
  plt.tight_layout()
  plt.savefig('fig/populasi-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

  fk_melanjutkan = list(df_wiraswasta[df_wiraswasta['Fakultas']==var]['Program Studi'].unique())
  frek = df_wiraswasta[df_wiraswasta['Fakultas']==var]['Program Studi'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Alumni Melanjutkan Kuliah Sumber Biaya', size=16)
  for i,k in enumerate(frek.values):
    plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
    plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.1,k/2), fontsize=12)
  plt.xticks(fk_melanjutkan, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,frek[0]+10)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

def sebaran_sektor_pekerjaan_wiraswasta_fakultas(df_wiraswasta, var, fontsz, figsz): # LEVEL FAKULTAS
  fig, ax =plt.subplots(1,figsize=figsz, frameon=False)
  # plt.title(var, size=16)
  plt.xticks([],[])
  plt.yticks(rotation = 30, fontsize=fontsz)
  ax.barh(df_wiraswasta.keys(), [df_wiraswasta[k] for k in df_wiraswasta.keys()], color='lightcoral', edgecolor='black')
  _=[ax.annotate('{}% ({})'.format(round(k/sum(df_wiraswasta.values)*100,2), k), (k+0.01,i), fontsize=10) for i,k in enumerate(df_wiraswasta.values)]
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/sebaran-sektor-pekerjaan-wiraswasta-fak-{}.png'.format(var), dpi=100, transparent=True)

def sebaran_kesesuaian_pendidikan_wiraswasta_fakultas(df_wiraswasta, var):
  tag='Tingkat pendidikan apa yang paling tepat/sesuai dengan pekerjaan anda saat ini ? (F15).1'
  datavar = df_wiraswasta[df_wiraswasta['Fakultas']==var][tag].value_counts()
  # order = [1,0,2,3]
  fig, ax=plt.subplots(figsize=(20,10))
  ax.barh(datavar.keys(), datavar.values, color='slateblue', edgecolor='black')
  plt.yticks(rotation = 40, fontsize=13)
  plt.xticks(rotation = 40, fontsize=13)
  _=[ax.annotate('  {}%'.format(round(datavar.values[k]/sum(datavar.values)*100)), (datavar.values[k],k), fontsize=14) for k in range(len(datavar))]
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-fak-{}.png'.format(var), dpi=100, transparent=True)

  tag2 = 'Jika pekerjaan saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya ? (F16).1'
  frek_pertanyan = df_wiraswasta[df_wiraswasta['Fakultas']==var][tag2].value_counts()
  ques = ['Q{}'.format(k) for k in range(1,len(frek_pertanyan))]
  ques2 = ['Q1: Prospek karir lebih baik', 'Q2: Belum mendapat yang lebih sesuai', 'Q3:Lebih menarik', 'Q4:Lebih menyukai karir saat ini', 'Q5:Pengalaman kerja', 'Q6: Lebih terjamin', 'Q7: Lokasi lebih dekat rumah', 'Q8:Pendapatan tinggi','Q9:Waktu lebih fleksibel', 'Q10:Dipromosikan', 'Q11:Lebih menjamin kebutuhan keluarga.']
  fig, ax=plt.subplots(figsize=(20,10))
  ax.bar(np.arange(len(ques)), frek_pertanyan.values[1:], edgecolor='black', color='blue')
  xmin, xmax, ymin, ymax = ax.axis()
  _ = [ax.annotate(k, ((xmax+xmin)/4,ymax-(i+1)*(ymax-ymin)/20)) for i,(k,c) in enumerate(zip(ques2,frek_pertanyan.values[1:]))]
  plt.xticks(np.arange(len(ques)),ques)
  plt.tight_layout()
  plt.savefig('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-fak-{}.png'.format(var), dpi=100, transparent=True)

def sebaran_gaji_wiraswasta_fakultas(df_wiraswasta, var, fontsz): # LEVEL FAKULTAS
  tag = 'Berapa pendapatan yang anda peroleh.1'
  salary = df_wiraswasta[df_wiraswasta['Fakultas']==var][tag].value_counts()
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
  plt.savefig('fig/sebaran-gaji-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

def sebaran_pendapatan_wiraswasta_fakultas(df_wiraswasta, var):
  prodi_sumberbiaya = df_wiraswasta[df_wiraswasta['Fakultas']==var]['Program Studi']
  pstudi_wiraswasta = list(prodi_sumberbiaya.unique())
  # print(len(pstudi_wiraswasta))

  term ='Berapa pendapatan yang anda peroleh.1'
  data3=df_wiraswasta[df_wiraswasta[term].notna()].copy()
  ### Per program Studi'
  # for fak in fakultas.keys():
  # prodi =[]
  # categ=['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000']
  sal = [4000000, 5500000, 8500000, 12500000, 20000000]

  # code = [k for k in range(5)]
  # codval =dict(zip(categ,code))

  mapping_gaji = {"< 4.000.000": 4000000,
                "4.000.000 - 7.000.000": 5500000,
                "7.000.000 - 10.000.000": 8500000,
                "10.000.000 - 15.000.000": 12500000,
                ">15.000.000": 20000000}
  data3[term] = data3[term].map(mapping_gaji)
  data3[term] 

  val =[]
  total = 0
  for i,k in enumerate(pstudi_wiraswasta):
    temp=data3.loc[data3['Program Studi']==k,term]

    val = val +[[temp.mean(),np.std(temp),temp.max()]]
    # print(val)
    # total=total+round(np.sum(salv*frek[1]/np.sum(frek[1])),0)
  # val = [val[k] for k in range(1,len(

  fig, ax = plt.subplots(1, figsize=(10,6))

  bar = ax.bar(pstudi_wiraswasta,[v[2] for v in val]) # x: rpodi y:frekuensi
  plt.xticks(rotation=90)
  plt.yticks([])

  def gradientbars(bars):
      grad = np.atleast_2d(np.linspace(0,1,256)).T
      ax = bars[0].axes
      lim = ax.get_xlim()+ax.get_ylim()
      for bar in bars:
          bar.set_zorder(1)
          bar.set_facecolor("none")
          x,y = bar.get_xy()
          w, h = bar.get_width(), bar.get_height()
          ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0)
      ax.axis(lim)
      ax.set_yticks(sal)
      ax.set_ylim(0,25000000)
      ax.set_xlim(-1,len(pstudi_wiraswasta))
      for i,vl in enumerate(val):
        ax.annotate(round(vl[0]/1000000,1), (float(i)-0.1,vl[0]), fontsize=12)
        ax.annotate(round(vl[1]/1000000,1), (float(i)-0.1,22000000), fontsize=14)
      # plt.annotate('mean: ', (-2,24000000), fontsize=14)
      plt.annotate('st.dev: ', (-0.9,22000000), fontsize=14)
      plt.annotate(' (juta)', (len(bars)-0.5,22000000), fontsize=14)

  gradientbars(bar)
  ax.set_ylabel('Nominal (Rupiah)')
  plt.title('Rata-rata gaji wiraswasta Fakultas {}'.format(var))
  xx, locs = plt.yticks()
  ll = ['%.0f' % a for a in xx]
  plt.yticks(xx, ll)
  # plt.show()
  plt.savefig('fig/gaji-wiraswasta-fak-{}.png'.format(var), dpi=200, transparent=True, bbox_inches='tight')
  # print(ll)
  # data3[term]

def sebaran_posisi_jabatan_wiraswasta_fakultas(df_wiraswasta, var, fontsz): # LEVEL FAKULTAS
  fig, ax = plt.subplots(figsize=(8,6))
  tag='Posisi/jabatan anda saat ini (F5c).1'
  posisi=df_wiraswasta[df_wiraswasta['Fakultas']==var][tag].value_counts()
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  ax.set_yticklabels(posisi.keys(),fontsize=12)
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100,1)),(k,i), fontsize=fontsz) for i,k in enumerate(posisi.values)]
  plt.tight_layout()
  plt.savefig('fig/sebaran-posisi-jabatan-wiraswasta-fak-{}.png'.format(var), dpi=100, transparent=True)

def waktu_tunggu_wiraswasta_fakultas(df_wiraswasta, var): # LEVEL FAKULTAS
  fig, ax = plt.subplots(1,figsize=(8,6), dpi=100)
  prodi = df_wiraswasta[df_wiraswasta['Fakultas']==var]['Program Studi']
  x = list(prodi.unique())
  
  y = [np.array(df_wiraswasta.loc[df_wiraswasta['Program Studi']==k,'Kapan anda memulai bekerja.1']).mean() for k in x]
  y_error = [np.array(df_wiraswasta.loc[df_wiraswasta['Program Studi']==k,'Kapan anda memulai bekerja.1']).std() for k in x]

  ax.errorbar(x, y, linestyle="None", yerr = y_error, fmt="or", markersize=9, capsize=3, ecolor="b", elinewidth=3)
  ax.set_ylim(-2.5,30)
  ax.set_xlim(-0.7,len(x))
  # ax.set_title('Rata-rata waktu tunggu bekerja lulusan')
  # ax.annotate('', (?,12.5))
  [ax.annotate('{}'.format(round(k,1)),(i,25.5)) for (i,k) in zip(x,y)]
  [ax.annotate('{}'.format(round(k,1)),(i,23.5)) for (i,k) in zip(x,y_error)]
  ax.set_ylabel('bulan')
  plt.xticks(rotation = 90)
  plt.tight_layout()
  plt.annotate('mean: ', (-0.5,25.5))
  plt.annotate('st.dev: ', (-0.5,23.5))

  ax.axhline(y = 10, color = 'g', linestyle = '-.', lw=1, label='max. 10 bulan')
  plt.legend(loc='center left')
  plt.savefig('fig/waktu-tunggu-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

def filtering_pdloc_wiraswasta_fakultas(df_wiraswasta, var):
  tag ='Di Kota/Kabupaten mana lokasi tempat usaha Anda?'
  frek = df_wiraswasta[(df_wiraswasta['Daerah Valid'] == 1) & (df_wiraswasta['Fakultas'] == var)][tag].value_counts()

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

def tempat_wiraswasta_fakultas(df_wiraswasta,dloc,var): #LEVEL PRODI
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

  # Add a point per position
  m.scatter(
      x,
      y,
      s=(dloc['pop']-dloc['pop'].min())/(dloc['pop'].max()-dloc['pop'].min())*10000,
      alpha=0.8,
      c=dloc['labels_enc']*100,
      cmap="Set1",
      zorder=5
  )
  # plt.show()
  ax2 = fig.add_axes([0.705, 0.51, 0.18, 0.18])
  province = df_wiraswasta[(df_wiraswasta['Daerah Valid'] == 1) & (df_wiraswasta['Fakultas'] == var)][tag].value_counts()
  province=province[province.values>=20]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  plt.xlim(0,400)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.savefig('fig/sebaran-lokasi-perusahaan-wiraswasta-fak-{}.png'.format(var), bbox_inches='tight', dpi=300, transparent=True)

def status_perusahaan_wiraswasta_fakultas(df_wiraswasta, fakultas, var): # LEVEL FAKULTAS
  col=['b', 'g', 'c', 'm', 'blueviolet', 'slateblue', 'lightcoral', 'brown']
  wiraswasta1 = list(df_wiraswasta['Apa status usaha anda?'].unique())
  width = 0.7
  frek = []
  for k in fakultas[var]:
    frek = frek +[df_wiraswasta[df_wiraswasta['Program Studi']==k]['Apa status usaha anda?'].value_counts()]
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  n=len(frek)
  # print(x)
  rects = [ax.bar((np.arange(len(frek[k]))-width/n*(n-1))+k*width/n,frek[k].values, width/(n+0.2), label=fakultas[var][k], color=col[k]) for k in range(n)]
  # [ax.annotate() for k in frek[n]]
  for i,k in enumerate(fakultas[var]):
    x = np.arange(len(frek[i]))-width/n*(n-1)
    [ax.annotate('{} %'.format(round(p/sum(frek[i].values)*100)), (x[m]+(i-0.2)*width/n,p+0.2), fontsize=11) for m,p in enumerate(frek[i].values)]

  plt.xticks(np.arange(len(wiraswasta1))-0.3, wiraswasta1)
  ax.legend()
  plt.tight_layout()
  # _=[ax.bar_label(k, padding=3) for k in rects]
  # ax.bar_label(rects2, padding=3)
  # print(frek[0], type(frek[0]))
  plt.savefig('fig/perusahaan-hukum-nonhukum-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

  wiraswasta2 = list(df_wiraswasta['Apa tingkat tempat usaha/perusahaan anda'].unique())
  width = 0.7
  frek = []
  for k in fakultas[var]:
    frek = frek +[df_wiraswasta[df_wiraswasta['Program Studi']==k]['Apa tingkat tempat usaha/perusahaan anda'].value_counts()]
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  n=len(frek)
  # print(x)
  rects = [ax.bar((np.arange(len(frek[k]))-width/n*(n-1))+k*width/n,frek[k].values, width/(n+0.2), label=fakultas[var][k], color=col[k]) for k in range(n)]
  # [ax.annotate() for k in frek[n]]
  for i,k in enumerate(fakultas[var]):
    x = np.arange(len(frek[i]))-width/n*(n-1)
    [ax.annotate('{} %'.format(round(p/sum(frek[i].values)*100)), (x[m]+(i-0.2)*width/n,p+0.2), fontsize=11) for m,p in enumerate(frek[i].values)]

  plt.xticks(np.arange(len(wiraswasta2))-0.3, wiraswasta2)
  ax.legend()
  plt.tight_layout()
  # _=[ax.bar_label(k, padding=3) for k in rects]
  # ax.bar_label(rects2, padding=3)
  # print(frek[0], type(frek[0]))
  plt.savefig('fig/perusahaan-nasional-multinasional-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

def top_perusahaan_wiraswasta_fakultas(df_wiraswasta, var): # LEVEL FAKULTAS
  tag1='Posisi/jabatan anda saat ini (F5c).1'
  tag2='Nama Perusahaan/Usaha Anda ? (F5b)'
  comment_words = ''
  stopwords = set(STOPWORDS)
  df = df_wiraswasta[df_wiraswasta['Fakultas']==var][tag2]

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
  plt.savefig('fig/top-perusahaan-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)

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
  _ = [ax.annotate('{}%'.format(round(k/sum(ls)*100)),(k,i), fontsize=12) for i,k in enumerate(ls)]
  plt.tight_layout()
  plt.savefig('fig/10-perusahaan-wiraswasta-fak-{}.png'.format(var), dpi=300, transparent=True)