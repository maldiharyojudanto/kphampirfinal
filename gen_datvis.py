import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from math import ceil
from mpl_toolkits.basemap import Basemap, maskoceans
from geopy.geocoders import Nominatim # library terusan geopy

import streamlit as st
from wordcloud import WordCloud, STOPWORDS
# from geopy.geocoders import Nominatim
# from mpl_toolkits.basemap import Basemap, maskoceans

def respon_rate(data, pop_dict):
    labl=['Merespon', 'Tidak merespon']
    val = [len(data), sum(pop_dict.values())-len(data)]
    pval = [round(v/sum(val)*100,1) for v in val]
    explode=(0, 0.05)
    print(pval)
    plt.subplot(1, 1, 1)
    plt.pie(val,labels=labl, autopct='%1.1f%%',
            shadow=True, startangle=90, explode=explode, pctdistance=0.6, labeldistance=1.25)
    plt.annotate('Populasi= {}'.format(int(sum(pop_dict.values()))), (-1.2,1.1))
    plt.savefig('fig/respon_rate_univ.png', bbox_inches='tight', dpi=300, transparent=True)

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_xdata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_xdata()[1]
        dict1['median'] = bp['medians'][i].get_xdata()[1]
        dict1['means'] = bp['means'][i].get_xdata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_xdata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_xdata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)

def plot_overall(data2, data):
  popu = dict(zip(['Universitas', 'D3', 'D4', 'S1', 'S2'], [13346,2039,163,10677,467]))
  pstudi=data2['Program Studi'].unique()
  wkt_tunggu =np.array(data2['Kapan anda memulai bekerja'])
  # wkt_tunggu.reshape(-1)

  wkt_dict = {"Universitas": wkt_tunggu,
              "D3":np.array(data2[data2['Program Studi'].str.startswith('D3')]['Kapan anda memulai bekerja']),
              "D4":np.array(data2[data2['Program Studi'].str.startswith('D4')]['Kapan anda memulai bekerja']),
              'S1':np.array(data2[data2['Program Studi'].str.startswith('S1')]['Kapan anda memulai bekerja']),
              'S2':np.array(data2[data2['Program Studi'].str.startswith('S2')]['Kapan anda memulai bekerja'])}
  wkt_dict2 = {"Universitas": wkt_tunggu,
              "2011":np.array(data2[data2['NIM'].str.startswith('11')]['Kapan anda memulai bekerja']),
              "2012":np.array(data2[data2['NIM'].str.startswith('12')]['Kapan anda memulai bekerja']),
              "2013":np.array(data2[data2['NIM'].str.startswith('13')]['Kapan anda memulai bekerja']),
              "2014":np.array(data2[data2['NIM'].str.startswith('14')]['Kapan anda memulai bekerja']),
              '2015':np.array(data2[data2['NIM'].str.startswith('15')]['Kapan anda memulai bekerja']),
              '2016':np.array(data2[data2['NIM'].str.startswith('16')]['Kapan anda memulai bekerja'])}
  wkt = {"Universitas": wkt_tunggu,
              "D3":np.array(data[data['Program Studi'].str.startswith('D3')]['Kapan anda memulai bekerja']),
              "D4":np.array(data[data['Program Studi'].str.startswith('D4')]['Kapan anda memulai bekerja']),
              'S1':np.array(data[data['Program Studi'].str.startswith('S1')]['Kapan anda memulai bekerja']),
              'S2':np.array(data[data['Program Studi'].str.startswith('S2')]['Kapan anda memulai bekerja'])}

  fig, ax = plt.subplots(3,2,figsize=(14,18), dpi=100)
  # # Set general font size
  plt.rcParams['font.size'] = '11'
  # # Set tick font size
  # # for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()+ax[1].get_xticklabels() + ax[1].get_yticklabels()):
  # # 	label.set_fontsize(14)
  expon = stats.gamma
  param = expon.fit(wkt_tunggu)
  labels = list(wkt_dict.keys())
  # print(labels, type(labels))
  positions = np.arange(5) + 1
  bp = ax[0,0].boxplot(wkt_dict.values(), vert=False, showfliers=True, labels =list(wkt_dict.keys()),widths=0.9, showmeans=True, meanline=True)
  bp2 = ax[1,0].boxplot(wkt_dict2.values(), vert=False, showfliers=True, labels =list(wkt_dict2.keys()),widths=0.9, showmeans=True, meanline=True)
  n, bins, patches = ax[2,0].hist(wkt_tunggu, color="#ff9999", width=1.5, bins=10,  edgecolor="black")
  ax[0,0].set_xlim(-0.2,np.max(wkt_tunggu)+15)
  ax[1,0].set_xlim(-0.2,np.max(wkt_tunggu)+15)
  ax[0,0].set_xlabel('bulan')
  jumlah = [len(wkt_dict[k]) for i,k in enumerate(list(wkt_dict.keys()))]
  jumlah2 = [len(wkt_dict2[k]) for i,k in enumerate(list(wkt_dict2.keys()))]
  ax[0,0].annotate('Jumlah resp.', (np.max(wkt_tunggu)+4,len(jumlah)+0.3))
  dbp = get_box_plot_data(list(wkt_dict.keys()), bp)
  _ = [ax[0,0].annotate('{}'.format(jumlah[i]), (np.max(wkt_tunggu)+3, i+0.8)) for i,k in enumerate(list(wkt_dict.keys()))]
  _ = [ax[0,0].annotate('{}'.format(round(np.mean(wkt_dict[k]),2)), (np.mean(wkt_dict[k]), i+1.2)) for i,k in enumerate(list(wkt_dict.keys()))]
  _ = [ax[0,0].annotate('{}'.format(round(np.median(wkt_dict[k]),2)), (np.median(wkt_dict[k]), i+0.8)) for i,k in enumerate(list(wkt_dict.keys()))]
  # _ = [ax[0,0].annotate('{}; {}% pencilan'.format(round(dbp.loc[i,'upper_whisker'],1),
  #                                                 round(len([elem for elem in list(wkt_dict.values())[i] if elem > dbp.loc[i,'upper_whisker']])/len(list(wkt_dict.values())[i])*100),1),
  #                       (dbp.loc[i,'upper_whisker']-0.5, i+1.3), fontsize=10) for i in range(len(dbp))]
  ax[2,0].set_xlabel('bulan')
  # ax[2,0].set_xticks(np.arange(0, np.max(wkt_tunggu)))
  # ax[0,0].set_facecolor('gainsboro')
  # ax[1,0].set_facecolor('gainsboro')
  dbp2 = get_box_plot_data(list(wkt_dict2.keys()), bp2)
  _ = [ax[1,0].annotate('{}'.format(jumlah2[i]), (np.max(wkt_tunggu)+3, i+0.8)) for i,k in enumerate(list(wkt_dict2.keys()))]
  _ = [ax[1,0].annotate('{}'.format(round(np.mean(wkt_dict2[k]),2)), (np.mean(wkt_dict2[k]), i+1.2)) for i,k in enumerate(list(wkt_dict2.keys()))]
  _ = [ax[1,0].annotate('{}'.format(round(np.median(wkt_dict2[k]),1)), (np.median(wkt_dict2[k]), i+0.8)) for i,k in enumerate(list(wkt_dict2.keys()))]
  # _ = [ax[1,0].annotate('{}; {}% pencilan'.format(round(dbp2.loc[i,'upper_whisker'],1),
                        #                           round(len([elem for elem in list(wkt_dict2.values())[i] if elem > dbp2.loc[i,'upper_whisker']])/len(list(wkt_dict2.values())[i])*100),1),
                        # (dbp2.loc[i,'upper_whisker']-0.5, i+1.3), fontsize=10) for i in range(len(dbp2))]
  # ax[2,0].set_facecolor('gainsboro')
  ax[2,0].annotate('Sebanyak {}% alumni \n memiliki masa tunggu kerja < 1,5 bulan'.format(round(len(wkt_tunggu[wkt_tunggu<1.5])/len(wkt_tunggu)*100,2)), (4,3400), fontsize=12)
  ax[2,0].annotate('Rata-rata waktu tunggu {} bulan'.format(round(np.mean(wkt_tunggu),2)),(4,2000) )
  ax[2,0].axvline(round(np.mean(wkt_tunggu),2), 0, 3000)
  labels = list(wkt_dict.keys())[1:]
  ax[0,1].pie([len(wkt_dict[k]) for k in labels],labels=labels, autopct='%1.1f%%',
          shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.25)
  # jumlah = [len(wkt[k]) for i,k in enumerate(list(wkt.keys()))]
  # jumlah2 = [len(wkt2[k]) for i,k in enumerate(list(wkt2.keys()))]
  jenjang = ['D3', 'D4', 'S1', 'S2']
  jumlah = [len(data[data['Program Studi'].str.startswith(k)]) for k in jenjang]
  jumlah = [sum(jumlah)] +jumlah
  cross_tab = pd.DataFrame({'index':list(wkt.keys()), 'merespon':jumlah,'tidak merespon':[popu[k]-jumlah[i] for i,k in enumerate(list(wkt_dict.keys()))]}).set_index('index')
  # axadd = fig.add_axes([0.3, 0.06, 0.2, 0.1])
  cross_tab.plot(kind='barh', stacked=True, colormap='tab10', ax=ax[2,1])
  lncr=len(cross_tab)
  _ = [ax[2,1].annotate('{}% (Pop. = {})'.format(round(cross_tab.loc[k,'merespon']/popu[k]*100,2),popu[k]), (cross_tab.loc[k,'merespon']+cross_tab.loc[k,'tidak merespon'], i)) for i,k in enumerate(list(wkt_dict.keys()))]
  ax[2,1].set_xticks([], [])
  ax[2,1].set_ylabel(None)
  ax[2,1].spines['top'].set_visible(False)
  ax[2,1].spines['right'].set_visible(False)
  ax[2,1].spines['bottom'].set_visible(False)
  ax[2,1].spines['left'].set_visible(False)
  # df_list[count].plot(axadd)
  ax[1,1].pie([len(wkt_dict2[k]) for k in list(wkt_dict2.keys())[1:]],labels=list(wkt_dict2.keys())[1:], autopct='%1.1f%%',shadow=True, startangle=90)
  xn =np.linspace(np.min(bins)-0.5,np.max(bins),100)
  y = expon.pdf(xn, *param)*len(wkt_tunggu)#interp1d(bins,np.histogram(wkt_tunggu,11)[0], kind='cubic')
  ax[2,0].plot(xn, y , linewidth=3, color='red')
  # plt.suptitle("Distribusi waktu tunggu lulusan bekerja", y=1, fontsize=14)
  plt.tight_layout()
  plt.savefig('fig/wkt_tunggu_univ.png', bbox_inches='tight', dpi=300, transparent=True)
  return xn, param, wkt_tunggu

# Waktu tunggu per prodi
def gen_wkt_tunggu(data2, fakultas):
  fig, ax = plt.subplots(1,figsize=(18,6), dpi=100)
  for fak in list(fakultas.keys()):
    # _ = [dis]
    x = list(fakultas[fak])
    y = [np.array(data2.loc[data2['Program Studi']==k,'Kapan anda memulai bekerja']).mean() for k in x]
    y_error = [np.array(data2.loc[data2['Program Studi']==k,'Kapan anda memulai bekerja']).std() for k in x]
    if fak=='FIF':
      print(y_error)

    ax.errorbar(x, y, linestyle="None", yerr = y_error, fmt="or", markersize=9, capsize=3, ecolor="b", elinewidth=3)
    ax.set_ylim(-2.5,16)
    ax.set_title('Rata-rata waktu tunggu bekerja lulusan')
    # ax.annotate('', (?,12.5))
    [ax.annotate('{}'.format(round(k,1)),(i,12.5)) for (i,k) in zip(x,y)]
    [ax.annotate('{}'.format(round(k,1)),(i,10.5)) for (i,k) in zip(x,y_error)]
    ax.set_ylabel('bulan')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.annotate('mean: ', (-1.2,12.5))
    plt.annotate('st.dev: ', (-1.2,10.5))

  ax.axhline(y = 4, color = 'g', linestyle = '-.', lw=1, label='max. 4 bulan')
  plt.legend(loc='center left')
  plt.savefig('fig/waktu-tunggu-semua-prodi.png', dpi=300, transparent=True)

# Tingkat Perusahaan Universitas

def tingkat_perusahaan_univ(df_usaha):
  bisnis = list(df_usaha['Jenis perusahaan'].unique())
  x = np.arange(len(bisnis))
  width = 0.2
  frek = df_usaha['Jenis perusahaan'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  [plt.annotate('{}'.format(k), (i-0.1,k+200), fontsize=12) for i,k in enumerate(frek.values)]
  [plt.annotate('{}%'.format(round(k/sum(frek.values)*100,2)), (i-0.2,k/2), fontsize=12) for i,k in enumerate(frek.values)]
  plt.xticks(bisnis, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,4000)
  plt.tight_layout()
  plt.savefig('fig/tingkat_perusahaan_univ.png', dpi=300, transparent=True)
  print(bisnis[-1])

# Sektor Perusahaan Universitas

def sektor_perusahaan_univ(df_usaha):
  tag = 'Sektor tempat anda bekerja'
  sektor = list(df_usaha[tag].unique())
  x = np.arange(len(sektor))
  width = 0.75
  frek = df_usaha[tag].value_counts()
  fig, ax = plt.subplots(1,1, figsize=(20,10))
  plt.title('Sektor tempat bekerja level universitas')
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # [plt.annotate('{}'.format(k), (i-0.1,k+200), fontsize=11) for i,k in enumerate(frek.values)]
  [plt.annotate('{}% ({})'.format(round(k/sum(frek.values)*100,2),k), (i,k+15), fontsize=11, rotation=90) for i,k in enumerate(frek.values)]
  plt.xticks(sektor, rotation=90, fontsize=11)
  plt.yticks(fontsize=11)
  plt.ylim(0,1000)
  plt.tight_layout()
  plt.savefig('fig/sektor_perusahaan_univ.png')

# Pendapatan Universitas

def gen_pendapatan(data2, fakultas):
  term ='Berapa pendapatan yang anda peroleh'
  data3=data2[data2[term].notna()].copy()
  ### Per program Studi'
  # for fak in fakultas.keys():
  sal = [4000000, 5500000, 8500000, 12500000, 20000000]
  prodi =[]
  for k in list(fakultas.values()):
    prodi = prodi +k

  mapping_gaji = {"< 4.000.000": 4000000,
                "4.000.000 - 7.000.000": 5500000,
                "7.000.000 - 10.000.000": 8500000,
                "10.000.000 - 15.000.000": 12500000,
                ">15.000.000": 20000000}
  data3[term] = data3[term].map(mapping_gaji)

  val =[]
  total = 0
  for i,k in enumerate(prodi):
    temp=data3.loc[data3['Program Studi']==k,term]

    val = val +[[temp.mean(),np.std(temp),temp.max()]]


  fig, ax = plt.subplots(1, figsize=(20,6))

  bar = ax.bar(prodi,[v[2] for v in val]) # x: rpodi y:frekuensi
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
      for i,vl in enumerate(val):
        ax.annotate(round(vl[0]/1000000,1), (float(i)-0.3,vl[0]), fontsize=12)
        ax.annotate(round(vl[1]/1000000,1), (float(i)-0.3,22000000), fontsize=14)
      # plt.annotate('mean: ', (-2,24000000), fontsize=14)
      plt.annotate('st.dev: ', (-2,22000000), fontsize=14)
      plt.annotate(' (juta)', (len(bars)-0.5,22000000), fontsize=14)

  gradientbars(bar)
  ax.set_ylabel('Nominal (Rupiah)')
  plt.title('Rata-rata gaji per Program Studi Tahun 2022')
  xx, locs = plt.yticks()
  ll = ['%.0f' % a for a in xx]
  plt.yticks(xx, ll)
  # plt.show()
  plt.savefig('fig/gaji-semua-prodi.png', dpi=200, transparent=True, bbox_inches='tight')
  # data3[term]

def salary_an(fak, data, data2):
  sal_str='Berapa pendapatan yang anda peroleh'
  plt.figure(figsize=[8, 8] ,dpi=100)
  salary = np.array(data['Berapa pendapatan yang anda peroleh'])
  salary_dict = {"Universitas": salary,
              "D3":np.array(data2[data2['Program Studi'].str.startswith('D3')][sal_str]),
              "D4":np.array(data2[data2['Program Studi'].str.startswith('D4')][sal_str]),
              'S1':np.array(data2[data2['Program Studi'].str.startswith('S1')][sal_str]),
              'S2':np.array(data2[data2['Program Studi'].str.startswith('S2')][sal_str])}
  salary_dict2 = {"Universitas": salary,
              "2011":np.array(data2[data2['NIM'].str.startswith('11')][sal_str]),
              "2012":np.array(data2[data2['NIM'].str.startswith('12')][sal_str]),
              "2013":np.array(data2[data2['NIM'].str.startswith('13')][sal_str]),
              "2014":np.array(data2[data2['NIM'].str.startswith('14')][sal_str]),
              '2015':np.array(data2[data2['NIM'].str.startswith('15')][sal_str]),
              '2016':np.array(data2[data2['NIM'].str.startswith('16')][sal_str])}
  labels=['< 4.000.000', '4.000.000 - 7.000.000', '7.000.000 - 10.000.000', '10.000.000 - 15.000.000', '>15.000.000']
  midpoint = np.array([4000000, 5500000, 8500000, 12500000, 20000000])
  # a, b, c = labels.index('4.000.000 - 7.000.000'), labels.index('7.000.000 - 10.000.000'),labels.index('< 4.000.000')
  # labels[a], labels[b], labels[c] = labels[c], labels[a], labels[b]
  # salary = salary.cat.codes.values
  plt.subplot(1,1,1, facecolor='gainsboro')
  plt.xticks(rotation = 45, fontsize=12)
  frek = np.unique(salary_dict['Universitas'],return_counts=True)[1]
  plt.bar(labels, frek, color="#ff9999", edgecolor='black')
  # plt.vlines(sum([frek[k]*midpoint[k]/len(frek) for k in range(len(frek))]), 0, 4000)
  plt.xticks(labels, rotation=30, fontsize=8)
  # plt.vline()
  mean_gaji = sum([frek[k]*midpoint[k]/sum(frek) for k in range(len(frek))])
  std_gaji = np.sqrt(sum([frek[k]*(midpoint[k]**2.)/sum(frek) for k in range(len(frek))])-(mean_gaji**2))
  plt.annotate("Rata-rata total gaji= Rp.{},-".format(math.floor(mean_gaji)),(1.7,1200), fontsize=12)
  print(math.floor(mean_gaji))
  plt.annotate("Standard deviasi total gaji= Rp.{},-".format(math.floor(std_gaji)),(1.7,1100), fontsize=12)
  plt.savefig('fig/pendapatan_univ.png', dpi=300, transparent=True)
  
# Sebaran Lokasi Kerja
def sebaran_univ(pdloc, df1):
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
  m.drawcoastlines()
  m.fillcontinents(color='aquamarine')
  pdloc['labels_enc'] = pd.factorize(pdloc['name'])[0]

  # Add a point per position
  m.scatter(
      x,
      y,
      s=(pdloc['pop']-pdloc['pop'].min())/(pdloc['pop'].max()-pdloc['pop'].min())*10000,
      alpha=0.8,
      c=pdloc['labels_enc']*100,
      cmap="Set1",
      zorder=5
  )
  ax2 = fig.add_axes([0.705, 0.51, 0.18, 0.18])
  province = df1['Provinsi tempat bekerja'].value_counts()
  province=province[province.values>=20]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  plt.xlim(0,3000)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)

  plt.savefig('fig/sebaran_lokasi_kerja_univ.png', bbox_inches='tight', dpi=300, transparent=True)

# Top 10 Perusahaan
def top_perusahaan_univ(firm_ten, ls):
  fig,ax=plt.subplots(1,1)
  ax.barh(firm_ten,ls, color='salmon', edgecolor='black')
  _ = [ax.annotate('{}%'.format(round(k/sum(ls)*100)),(k,i), fontsize=12) for i,k in enumerate(ls)]
  plt.savefig('fig/top10perusahaan_univ.png', bbox_inches='tight', dpi=300, transparent=True)

# Jabatan Universitas
def jabatan_univ(df1, tag1, tag2):
  posisi=df1[tag1].value_counts()
  firm=df1[tag2].value_counts()
  fig,ax=plt.subplots(1,1)
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100)),(k,i), fontsize=12) for i,k in enumerate(posisi.values)]
  plt.savefig('fig/jabatan_univ.png', dpi=300, bbox_inches='tight')

# Status Pekerjaan Universitas
def addlabels(x,y, fnsz=12):
  for i in range(len(x)):
    plt.text(i, y[i]+20, y[i], ha = 'center', fontsize=fnsz)
    plt.text(i, y[i]/2, "{0:.0%}".format(round(y[i]/np.sum(np.array(y)),4)), ha = 'center', fontsize=fnsz)

def distribusi(a,fontsz, figsz, data):
  dataprovin = data[data[a].notnull()][a].value_counts()
  plt.figure(figsize=figsz, dpi=80)
  plt.xticks(rotation = 45, fontsize=fontsz)
  plt.yticks(rotation = 45, fontsize=fontsz)
  plt.bar(dataprovin.keys(), dataprovin.values)
  addlabels([str(i) for i in dataprovin.keys()],dataprovin.values, fontsz)

def status_pekerjaan_univ(data):
  distribusi('Status Anda saat ini (F8)', 14, (15,7), data)
  plt.tight_layout()
  plt.savefig('fig/status_pekerjaan_univ.png',dpi=300, transparent=True)

############################################ MELANJUTKAN PENDIDIKAN ######################################
def frekuensi_melanjutkan_pendidikan(data, df_melanjutkan): # LEVEL UNIVERSITAS
  labl = ['Melanjutkan Pendidikan', 'Semua Mahasiswa/Alumni']
  val = [len(df_melanjutkan), len(data)]
  print(val)
  # print(val)
  pval = [round(v/sum(val)*100,1) for v in val]
  explode = (0, 0.05) # jarak pada chart
  # print(pval)
  fig, ax =plt.subplots(figsize=(8,6)) # 1: respon 2: sektor 3:tingkat pend 4:jika tidak sesuai
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  plt.annotate('Populasi (Semua Responden) = {}'.format(len(data)), (-1.2,0.8))
  plt.tight_layout()
  plt.savefig('fig/populasi-melanjutkan-pendidikan-univ.png', dpi=300, transparent=True)

  sektorfak = list(df_melanjutkan['Fakultas'].unique())
  frek = df_melanjutkan['Fakultas'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Alumni Melanjutkan Kuliah Level Fakultas', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.2,k/2), fontsize=12)
  plt.xticks(sektorfak, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,300)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-melanjutkan-pendidikan-fakuniv.png', dpi=300, transparent=True)

  sektorprodi = list(df_melanjutkan['Program Studi'].unique())
  frek = df_melanjutkan['Program Studi'].value_counts()
  fig, ax = plt.subplots(1,1, figsize=(20,10))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Alumni Melanjutkan Studi Level Prodi', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+10), fontsize=11)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.3,k/2+5), fontsize=11)
  plt.xticks(sektorprodi, rotation=90, fontsize=11)
  plt.yticks(fontsize=11)
  plt.ylim(0,70)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-melanjutkan-pendidikan-prodiuniv.png', dpi=300, transparent=True)

def sumber_biaya_melanjutkan_pendidikan(df_melanjutkan): # LEVEL UNIVERSITAS
  sumberpendidikan = list(df_melanjutkan['Sumber Biaya'].unique())
  frek = df_melanjutkan['Sumber Biaya'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Alumni Melanjutkan Kuliah Sumber Biaya Level Universitas', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+20), fontsize=12)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.1,k/2), fontsize=12)
  plt.xticks(sumberpendidikan, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,600)
  plt.tight_layout()
  plt.savefig('fig/sumber-biaya-melanjutkan-univ.png', dpi=300, transparent=True)

def sebaran_melanjutkan_pendidikan(df_melanjutkan, tag1): # LEVEL UNIVERSITAS
  comment_words = ''
  stopwords = set(STOPWORDS)

  # iterate through the csv file
  for val in df_melanjutkan[tag1]:
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
  st.set_option('deprecation.showPyplotGlobalUse', False) # untuk menghilangkan warning bawaan dari streamlit
  plt.tight_layout()
  plt.savefig('fig/sebaran-melanjutkan-pendidikan-univ.png', dpi=300, transparent=True)

    # print(type(text))
  # print(text)
  wordcloud.words_.keys()
  term_univ =["Telkom University|telkom university|Universitas Telkom|university telkom|universitas telkom|UNIV TELKOM|TELKOM UNIVERSITY|Universitas telkom", "Universitas Indonesia", "Universitas Widyatama|Universitas Widiatama", "ITB|itb|institut teknologi bandung|Institut Teknologi Bandung|Institut Teknologi Bandung (ITB)", 'Bina Nusantara|Binus|bina nusantara|binus|Binus University', "Universitas Diponegoro", 'Padjajaran|Universitas Padjajaran|padjajaran|Universitas Padjadjaran','UGM|ugm|universitas gadjah mada|Universitas Gadjah Mada|UNIVERSITAS GAJAH MADA', "Institut Pertanian Bogor|IPB"]
  firm_ten_univ=['Telkom University', "Universitas Indonesia", "Universitas Widyatama", 'ITB', 'BINUS', 'Universitas Diponegoro', 'Universitas Padjajaran', 'UGM', 'IPB']
  ls = [df_melanjutkan[tag1].str.count(k).sum() for k in term_univ]
  # print(len(ls),ls)
  # print(ls)
  firm_ten_univ=[i for k,i in sorted(zip(ls,firm_ten_univ))]
  ls.sort()
  # print(firm_ten_univ, ls)
  
  fig, ax=plt.subplots(figsize=(8,6))
  ax.barh(firm_ten_univ,ls, color='salmon', edgecolor='black')
  ax.set_yticklabels(firm_ten_univ,fontsize=12)
  _ = [ax.annotate('{}%'.format(round(k/sum(ls)*100)),(k,i), fontsize=12) for i,k in enumerate(ls)]
  plt.tight_layout()
  plt.savefig('fig/top-universitas-melanjutkan-univ.png', dpi=300, transparent=True)

  comment_words = ''
  stopwords = set(STOPWORDS)

  # iterate through the csv file
  for val in df_melanjutkan.iloc[:, 81]:
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
  st.set_option('deprecation.showPyplotGlobalUse', False) # untuk menghilangkan warning bawaan dari streamlit
  plt.tight_layout()
  plt.savefig('fig/sebaran-melanjutkan-pendidikan-prodi.png', dpi=300, transparent=True)

      # print(type(text))
  # print(text)
  wordcloud.words_.keys()
  term_prodi =["Akuntansi|S1 Akuntansi", "Teknik Telekomunikasi|S1 Teknik Telekomunikasi|S2 TEKNIK TELEKOMUNIKASI", "Administrasi Bisnis|S1 Administrasi Bisnis|Master of Business Administration|MBA|Master Of Business Administration", "Sistem Informasi|S1 Sistem Informasi|S1 SISTEM INFORMASI", 'Informatika|S1 Informatika|Teknik Informatika|S2 Informatika|S1 Teknik Informatika|S2 INFORMATIKA', "Manajemen|Magister Manajemen|S1 Manajemen|S2 Manajemen|Management", 'Teknik Komputer', 'Magister Desain|S2 Desain', 'Teknik Elektro|S2 Teknik Elektro', 'Teknik Industri|S2 Teknik Industri', 'Magister Ilmu Komunikasi|S2 ILMU KOMUNIKASI']
  firm_ten_prodi=["Akuntansi", "Teknik Telekomunikasi", "Administrasi Bisnis", "Sistem Informasi", 'Informatika', "Manajemen", 'Teknik Komputer', 'Magister Desain', 'Teknik Elektro', 'Teknik Industri', 'Ilmu Komunikasi']
  ls = [df_melanjutkan.iloc[:, 81].str.count(k).sum() for k in term_prodi]
  # print(len(ls),ls)
  # print(ls)
  firm_ten_prodi=[i for k,i in sorted(zip(ls,firm_ten_prodi))]
  ls.sort()
  # print(firm_ten_prodi, ls)
  
  fig, ax=plt.subplots(figsize=(8,6))
  ax.barh(firm_ten_prodi,ls, color='salmon', edgecolor='black')
  ax.set_yticklabels(firm_ten_prodi,fontsize=12)
  _ = [ax.annotate('{}%'.format(round(k/sum(ls)*100)),(k,i), fontsize=12) for i,k in enumerate(ls)]
  plt.tight_layout()
  plt.savefig('fig/top-prodi-melanjutkan-prodi.png', dpi=300, transparent=True)

  
#########################################################################################################

################################################ TIDAK BEKERJA ############################################
def frekuensi_tidak_bekerja(data, df_tidakbekerja): # LEVEL UNIVERSITAS
  labl = ['Tidak Bekerja', 'Semua Mahasiswa/Alumni']
  val = [len(df_tidakbekerja), len(data)]
  print(val)
  # print(val)
  pval = [round(v/sum(val)*100,1) for v in val]
  explode = (0, 0.05) # jarak pada chart
  # print(pval)
  fig, ax =plt.subplots(figsize=(8,6)) # 1: respon 2: sektor 3:tingkat pend 4:jika tidak sesuai
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  plt.annotate('Populasi (Semua Responden) = {}'.format(len(data)), (-1.2,0.8))
  plt.tight_layout()
  plt.savefig('fig/populasi-tidak-bekerja-univ.png', dpi=300, transparent=True)

  sektorfak = list(df_tidakbekerja['Fakultas'].unique())
  frek = df_tidakbekerja['Fakultas'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Studi Wiraswasta Alumni Level Universitas', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.2,k/2), fontsize=12)
  plt.xticks(sektorfak, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,400)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-tidak-bekerja-fakuniv.png', dpi=300, transparent=True)

  sektorprodi = list(df_tidakbekerja['Program Studi'].unique())
  frek = df_tidakbekerja['Program Studi'].value_counts()
  fig, ax = plt.subplots(1,1, figsize=(20,10))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Alumni Tidak Bekerja Level Prodi', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+10), fontsize=11)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.3,k/2+5), fontsize=11)
  plt.xticks(sektorprodi, rotation=90, fontsize=11)
  plt.yticks(fontsize=11)
  plt.ylim(0,190)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-tidak-bekerja-prodiuniv.png', dpi=300, transparent=True)

def distribusirata(value, min, max): # fungsi untuk mencari distribusi rata-rata melanjutkan pendidikan
    distrib = (value-min)/(max-min)
    return distrib

def level_motivasi_tidak_bekerja(df_tidakbekerja): # LEVEL UNIVERSITAS

  df_tidakbekerja['Rata-rata (Banyak lamaran, Banyak perusahaan yang merespon lamaran, Berapa banyak undangan wawancara)'] = df_tidakbekerja.iloc[:, 113:116].mean(axis = 1, skipna = True)
  ratarata = df_tidakbekerja['Rata-rata (Banyak lamaran, Banyak perusahaan yang merespon lamaran, Berapa banyak undangan wawancara)']

  # bikin kolom untuk menampung (di-min(d))/(max(d)-min(d))
  distribusi = []
  for i,k in enumerate(ratarata):
      distribusi.append(distribusirata(k, ratarata.min(), ratarata.max()))

  df_tidakbekerja['Distribusi'] = distribusi
  df_tidakbekerja['Distribusi'] = df_tidakbekerja['Distribusi'].fillna(0) # supaya tidak terdeteksi nan/diganti menjadi 0

  # normalisasi distribusi
  normalisasi = [
      (df_tidakbekerja['Distribusi'] <= 0.1),
      (df_tidakbekerja['Distribusi'] > 0.1) & (df_tidakbekerja['Distribusi'] <= 0.2),
      (df_tidakbekerja['Distribusi'] > 0.2) & (df_tidakbekerja['Distribusi'] <= 0.5),
      (df_tidakbekerja['Distribusi'] > 0.5) & (df_tidakbekerja['Distribusi'] <= 1)]
  # label untuk 0.1 adalah 1, 0.1-0.2 adalah 2, 0.2-0.5 adalah 3, 0.5-1 adalah 4
  label = [1, 2, 3, 4]
  # terapkan label dengan kondisi diatas
  df_tidakbekerja['Distribusi'] = np.select(normalisasi, label)

  # mapping value kolom 'Apakah anda aktif mencari pekerjaan dalam 4 minggu terakhir ?' menjadi numerik
  mapping_aktifmencaripekerjaan = {"Tidak": 1,
              "Tidak, tapi saya sedang menunggu hasil lamaran kerja": 2,
              "Ya, tapi saya belum pasti akan bekerja dalam 2 minggu ke depan": 3,
              "Ya, saya akan mulai bekerja dalam 2 minggu ke depan": 4}
  df_tidakbekerja['Apakah anda aktif mencari pekerjaan dalam 4 minggu terakhir ? (F10)'] = df_tidakbekerja["Apakah anda aktif mencari pekerjaan dalam 4 minggu terakhir ? (F10)"].map(mapping_aktifmencaripekerjaan)

  # normalisasi tadi di tambahkan dengan alasan, dan dicari rata-ratanya (menggunakan round)
  normalisasi = round((df_tidakbekerja.iloc[:, 83]  + df_tidakbekerja.iloc[:, 118])/2)
  df_tidakbekerja['Distribusi Final (rata rata dari aktif mencari pekerjan dalam 4 minggu + distribusi yang telah dinormalisasi)'] = normalisasi

  # mapping value kolom '
  mapping_normalisasi = {1: "Sangat Rendah",
                          2: "Rendah",
                          3: "Cukup",
                          4: "Tinggi"}
  df_tidakbekerja['Distribusi Final (rata rata dari aktif mencari pekerjan dalam 4 minggu + distribusi yang telah dinormalisasi)'] = df_tidakbekerja['Distribusi Final (rata rata dari aktif mencari pekerjan dalam 4 minggu + distribusi yang telah dinormalisasi)'].map(mapping_normalisasi)

  datavar = df_tidakbekerja['Distribusi Final (rata rata dari aktif mencari pekerjan dalam 4 minggu + distribusi yang telah dinormalisasi)'].value_counts()
  order = [1,0,2,3]
  fig, ax=plt.subplots(figsize=(12,8))
  # plt.suptitle('Level Motivasi Tidak Bekerja',size=18)
  ax.barh([datavar.keys()[i] for i in order], [datavar.values[i] for i in order], color='slateblue', edgecolor='black')
  plt.yticks(rotation = 40, fontsize=13)
  plt.xticks(rotation = 40, fontsize=13)
  _=[ax.annotate('{}%'.format(round(datavar.values[k]/sum(datavar.values)*100)), (datavar.values[k]+5,i), fontsize=14) for i,k in enumerate(order)]
  for spine in ['top', 'right', 'left']:
      ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/level-motivasi-tidak-bekerja-univ.png', dpi=300, transparent=True)
#########################################################################################################

################################################ WIRASWASTA ############################################
def frekuensi_wiraswasta(data, df_wiraswasta): # LEVEL UNIVERSITAS
  labl = ['Wiraswasta', 'Semua Mahasiswa/Alumni']
  val = [len(df_wiraswasta), len(data)]
  print(val)
  # print(val)
  pval = [round(v/sum(val)*100,1) for v in val]
  explode = (0, 0.05) # jarak pada chart
  # print(pval)
  fig, ax =plt.subplots(figsize=(8,6)) # 1: respon 2: sektor 3:tingkat pend 4:jika tidak sesuai
  patches=ax.pie(val, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.6, radius=0.7)
  ax.legend(patches, labels=labl, loc="upper left")
  plt.annotate('Populasi (Semua Responden) = {}'.format(len(data)), (-1.2,0.8))
  plt.tight_layout()
  plt.savefig('fig/populasi-wiraswasta-univ.png', dpi=300, transparent=True)

  sektorfak = list(df_wiraswasta['Fakultas'].unique())
  frek = df_wiraswasta['Fakultas'].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(8,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Studi Wiraswasta Alumni Level Fakultas', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+5), fontsize=12)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.2,k/2), fontsize=12)
  plt.xticks(sektorfak, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,150)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-wiraswasta-fakuniv.png', dpi=300, transparent=True)

  sektorprodi = list(df_wiraswasta['Program Studi'].unique())
  frek = df_wiraswasta['Program Studi'].value_counts()
  fig, ax = plt.subplots(1,1, figsize=(20,10))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Frekuensi Studi Wiraswasta Alumni Level Prodi', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+10), fontsize=11)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.3,k/2+5), fontsize=11)
  plt.xticks(sektorprodi, rotation=90, fontsize=11)
  plt.yticks(fontsize=11)
  plt.ylim(0,85)
  plt.tight_layout()
  plt.savefig('fig/frekuensi-wiraswasta-prodiuniv.png', dpi=300, transparent=True)

def sebaran_sektor_pekerjaan_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  sektor = list(df_wiraswasta['Sektor tempat anda bekerja.1'].unique())
  frek = df_wiraswasta['Sektor tempat anda bekerja.1'].value_counts()
  fig, ax = plt.subplots(1,1, figsize=(20,10))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  # plt.title('Sektor Wiraswasta Alumni Universitas', size=16)
  for i,k in enumerate(frek.values):
      plt.annotate('{}'.format(k), (i-0.1,k+20), fontsize=11)
      plt.annotate('{}%'.format(round(k/sum(frek.values)*100,1)), (i-0.3,k/2+10), fontsize=11)
  plt.xticks(sektor, rotation=90, fontsize=11)
  plt.yticks(fontsize=11)
  plt.ylim(0,150)
  plt.tight_layout()
  plt.savefig('fig/sebaran-sektor-pekerjaan-wiraswasta-univ.png', dpi=300, transparent=True)

def sebaran_kesesuaian_pendidikan_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  frek_tingkat = df_wiraswasta['Tingkat pendidikan apa yang paling tepat/sesuai dengan pekerjaan anda saat ini ? (F15).1'].value_counts()
  order = [1,0,2,3]
  fig, ax=plt.subplots(figsize=(20,10))
  # plt.suptitle('Kesesuaian Pendidikan Alumni Wiraswasta Level Universitas dan Alasan Tidak Sesuai',size=18)
  ax.barh([frek_tingkat.keys()[i] for i in order], [frek_tingkat.values[i] for i in order], color='slateblue', edgecolor='black')
  plt.yticks(rotation = 40, fontsize=13)
  plt.xticks(rotation = 40, fontsize=13)
  _=[ax.annotate('{}%'.format(round(frek_tingkat.values[k]/sum(frek_tingkat.values)*100)), (frek_tingkat.values[k]+5,i), fontsize=14) for i,k in enumerate(order)]
  for spine in ['top', 'right', 'left']:
      ax.spines[spine].set_visible(False)
  plt.tight_layout()
  plt.savefig('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-univ.png', dpi=100, transparent=True)

  frek_pertanyan = df_wiraswasta['Jika pekerjaan saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya ? (F16).1'].value_counts()
  fig, ax=plt.subplots(figsize=(20,10))
  ques = ['Q{}'.format(k) for k in range(1,len(frek_pertanyan))]
  ques2 = ['Q1: Prospek karir lebih baik', 'Q2: Belum mendapat yang lebih sesuai', 'Q3: Lebih menarik', 'Q4: Lebih menyukai karir saat ini', 'Q5: Pengalaman kerja', 'Q6: Lebih terjamin', 'Q7: Lokasi lebih dekat rumah', 'Q8: Pendapatan tinggi','Q9: Waktu lebih fleksibel', 'Q10: Dipromosikan', 'Q11: Lebih menjamin kebutuhan keluarga.']
  ax.bar(np.arange(len(ques)), frek_pertanyan.values[1:], edgecolor='black', color='blue')
  xmin, xmax, ymin, ymax = ax.axis()
  _ = [ax.annotate(k, ((xmax+xmin)/1.5,ymax-(i+1)*(ymax-ymin)/20)) for i,(k,c) in enumerate(zip(ques2,frek_pertanyan.values[1:]))]
  plt.xticks(np.arange(len(ques)),ques)
  plt.tight_layout()
  plt.savefig('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-univ.png', dpi=100, transparent=True)

def sebaran_gaji_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  salary = df_wiraswasta['Berapa pendapatan yang anda peroleh.1'].value_counts()
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
  h=(ymax-ymin)/70
  ax.annotate("Standard deviasi gaji wiraswasta= Rp.{},-".format(math.floor(std_gaji)),(xmin+0.5*(xmax-xmin),ymax-0.3*(ymax-ymin)), fontsize=12)
  _ = [ax.annotate('{} %'.format(round(s/sum(salary.values)*100,1)),(k,s+h), fontsize=12) for (k,s) in zip([salary.keys()[j] for j in ref],[salary.values[k] for k in ref])]
  plt.tight_layout()
  plt.savefig('fig/sebaran-gaji-wiraswasta-univ.png', dpi=100, transparent=True)

  pstudi_wiraswasta = list(df_wiraswasta['Program Studi'].unique())
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

  fig, ax = plt.subplots(1, figsize=(20,6))

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
      for i,vl in enumerate(val):
        ax.annotate(round(vl[0]/1000000,1), (float(i)-0.3,vl[0]), fontsize=12)
        ax.annotate(round(vl[1]/1000000,1), (float(i)-0.3,22000000), fontsize=14)
      # plt.annotate('mean: ', (-2,24000000), fontsize=14)
      plt.annotate('st.dev: ', (-1.8,22000000), fontsize=14)
      plt.annotate(' (juta)', (len(bars)-0.5,22000000), fontsize=14)

  gradientbars(bar)
  ax.set_ylabel('Nominal (Rupiah)')
  plt.title('Rata-rata gaji per Program Studi Tahun 2022')
  xx, locs = plt.yticks()
  ll = ['%.0f' % a for a in xx]
  plt.yticks(xx, ll)
  # plt.show()
  plt.savefig('fig/gaji-wiraswasta-prodi.png', dpi=200, transparent=True, bbox_inches='tight')
  # print(ll)
  # data3[term]

def sebaran_posisi_jabatan_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  posisi=df_wiraswasta['Posisi/jabatan anda saat ini (F5c).1'].value_counts()
  fig, ax = plt.subplots(figsize=(8,6))
  # plt.suptitle('Sebaran Posisi dan Gaji Level Universitas', fontsize=16)
  ax.barh(posisi.keys(), posisi.values, color='salmon', edgecolor='black')
  ax.set_yticklabels(posisi.keys(),fontsize=12)
  _ = [ax.annotate('{}%'.format(round(k/sum(posisi.values)*100)),(k,i), fontsize=12) for i,k in enumerate(posisi.values)]
  plt.tight_layout()
  plt.savefig('fig/sebaran-posisi-jabatan-wiraswasta-univ.png', dpi=100, transparent=True)

def waktu_tunggu_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  fig, ax = plt.subplots(1,figsize=(18,6), dpi=100)
  prodi_wiraswasta = list(df_wiraswasta['Program Studi'].unique())

  # _ = [dis]
  x = list(prodi_wiraswasta)
  y = [np.array(df_wiraswasta.loc[df_wiraswasta['Program Studi']==k,'Kapan anda memulai bekerja.1']).mean() for k in x]
  y_error = [np.array(df_wiraswasta.loc[df_wiraswasta['Program Studi']==k,'Kapan anda memulai bekerja.1']).std() for k in x]

  ax.errorbar(x, y, linestyle="None", yerr = y_error, fmt="or", markersize=9, capsize=3, ecolor="b", elinewidth=3)
  ax.set_ylim(-2.5,30)
  # ax.set_title('Rata-rata waktu tunggu bekerja lulusan')
  # ax.annotate('', (?,12.5))
  [ax.annotate('{}'.format(round(k,1)),(i,25.5)) for (i,k) in zip(x,y)]
  [ax.annotate('{}'.format(round(k,1)),(i,23.5)) for (i,k) in zip(x,y_error)]
  ax.set_ylabel('bulan')
  plt.xticks(rotation = 90)
  plt.tight_layout()
  plt.annotate('mean: ', (-1.2,25.5))
  plt.annotate('st.dev: ', (-1.2,23.5))

  ax.axhline(y = 10, color = 'g', linestyle = '-.', lw=1, label='max. 10 bulan')
  plt.legend(loc='center left')
  plt.savefig('fig/waktu-tunggu-wiraswasta-univ.png', dpi=300, transparent=True)

def filtering_tempat_wiraswasta(df_wiraswasta): # LEVEL UNIVERSITAS
  tag ='Di Kota/Kabupaten mana lokasi tempat usaha Anda?'
  frek = df_wiraswasta[tag].value_counts()
  loc = list(frek.keys())

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

  locdict= {'nama':namakota, 'lat':lats, 'lon':lons}
  # print(len(locdict))
  pdloc=pd.DataFrame(locdict)
  # print(len(pdloc))
  # print(namakota)
  # locdict= {'name':frek.keys(),'pop':frek.values, 'lat':lats, 'lon':lons}
  # pdloc=pd.DataFrame(locdict)

  count = {}
  for i, k in enumerate(namakota):
    if k in frek:
      if k in count:
          count[k] += 1
      else:
          count[k] = 1
    else:
        print(i)
  df_wiraswasta['Daerah Valid'] = df_wiraswasta['Di Kota/Kabupaten mana lokasi tempat usaha Anda?'].map(count)
  return df_wiraswasta

def filtering_pdloc_wiraswasta(df_wiraswasta):
  tag ='Di Kota/Kabupaten mana lokasi tempat usaha Anda?'
  frek = df_wiraswasta[df_wiraswasta['Daerah Valid'] == 1][tag].value_counts()

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
  
def tempat_wiraswasta(df_wiraswasta,dloc): #LEVEL UNIV
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
  province = df_wiraswasta['Di Provinsi manakah perusahaan anda berada?'].value_counts()
  province=province[province.values>=20]
  print(type(province))
  ax2.barh(province.keys(),province.values, color='red')
  plt.yticks(fontsize=11)
  plt.xlim(0,400)
  _=[ax2.annotate('{}%'.format(round(k/sum(province.values)*100,1)),(k,i)) for i,k in enumerate(province.values)]
  for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
    ax.spines[spine].set_visible(False)
  plt.show()

def status_perusahaan_wiraswasta(df_wiraswasta, tag1, tag2): # LEVEL UNIVERSITAS
  hukumnon = list(df_wiraswasta[tag1].unique())
  frek = df_wiraswasta[tag1].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(6,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  for i,k in enumerate(frek.values):
    plt.annotate('{}'.format(k), (i-0.1,k+30), fontsize=12)
    plt.annotate('{}%'.format(round(k/sum(frek.values)*100,2)), (i-0.2,k/2), fontsize=12)
  plt.xticks(hukumnon, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,500)
  plt.tight_layout()
  plt.savefig('fig/perusahaan-hukum-nonhukum-wiraswasta-univ.png', dpi=300, transparent=True)

  nasionalnon = list(df_wiraswasta[tag2].unique())
  frek = df_wiraswasta[tag2].value_counts()
  fig, ax = plt.subplots(1,1,figsize=(6,6))
  plt.bar(frek.keys(),frek.values, color='lightcoral', edgecolor='black')
  for i,k in enumerate(frek.values):
    plt.annotate('{}'.format(k), (i-0.1,k+30), fontsize=12)
    plt.annotate('{}%'.format(round(k/sum(frek.values)*100,2)), (i-0.2,k/2), fontsize=12)
  plt.xticks(nasionalnon, rotation=20, fontsize=12)
  plt.yticks(fontsize=11)
  plt.ylim(0,500)
  plt.tight_layout()
  plt.savefig('fig/perusahaan-nasional-multinasional-wiraswasta-univ.png', dpi=300, transparent=True)

def top_perusahaan_wiraswasta(df_wiraswasta):
  tag1='Posisi/jabatan anda saat ini (F5c).1'
  tag2='Nama Perusahaan/Usaha Anda ? (F5b)'
  comment_words = ''
  stopwords = set(STOPWORDS)

  # iterate through the csv file
  for val in df_wiraswasta[tag2]:

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
  plt.savefig('fig/top-perusahaan-wiraswasta-univ.png', dpi=300, transparent=True)

  # print(type(text))
  # print(text)
  wordcloud.words_.keys()
  term_wirausaha =["PT|pt|CV|cv|Indonesia|indonesia|id|putra|mandiri", "Toko|toko|shop|store|Coffe|coffe", 'Freelance|freelance|lepas', 'Desain|desain|desain interior|studio|seniman|interior','Tour|tour|Travel|travel|Laundry|laundry']
  firm_ten_wirausaha=['Pengusaha', 'Pedagang', 'Pekerja Lepas', 'Desainer', 'Bidang Jasa']
  ls = [df_wiraswasta[tag2].str.count(k).sum() for k in term_wirausaha]
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
  plt.savefig('fig/10-perusahaan-wiraswasta-univ.png', dpi=300, transparent=True)