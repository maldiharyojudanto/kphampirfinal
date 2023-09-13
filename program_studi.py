import streamlit as st
from streamlit_option_menu import option_menu
import os
from gen_datvis import filtering_tempat_wiraswasta
from gen_datvis_prodi import filtering_pdloc_wiraswasta_prodi, tempat_wiraswasta_prodi

warnaProdi = {"S1 Informatika": "#B28D35","S1 Teknologi Informasi": "#B28D35", "S2 Informatika": "#B28D35",
            "S1 Teknik Industri": "#0B6623", "S1 Sistem Informasi": "#0B6623", "S2 Teknik Industri": "#0B6623",
            "S1 Teknik Telekomunikasi": "#000080", "S1 Teknik Elektro": "#000080", "S1 Teknik Komputer": "#000080", "S1 Teknik Fisika": "#000080", "S2 Teknik Elektro" : "#000080",
            "S1 International ICT Business":"#5959FF", "S2 Manajemen":"#5959FF", "S1 Akuntansi":"#5959FF", "S1 Manajemen":"#5959FF",
            "S1 Ilmu Komunikasi (Int.)": "#8510BC", "S1 Administrasi Bisnis (Int.)": "#8510BC", "S1 Hubungan Masyarakat": "#8510BC", "S1 Ilmu Komunikasi": "#8510BC", "S1 Administrasi Bisnis": "#8510BC",
            "S1 Seni Rupa": "#FF7F00", "S1 Desain Produk": "#FF7F00", "S1 Kriya": "#FF7F00", "S1 Desain Interior": "#FF7F00", "S1 Desain Komunikasi Visual": "#FF7F00",
            "D3 Sistem Informasi": "#3EB049", "D3 Teknologi Komputer": "#3EB049", "D3 Sistem Informasi Akuntansi": "#3EB049", "D3 Manajemen Pemasaran": "#3EB049", "D3 Teknologi Telekomunikasi": "#3EB049", "D3 Rekayasa Perangkat Lunak Aplikasi": "#3EB049", "D3 Perhotelan": "#3EB049", "D4 Teknologi Rekayasa Multimedia": "#3EB049"}


def custom_subheading(text, color):
    # Definisikan CSS inline untuk menyesuaikan tampilan teks
    style = f"""
        color: white;
        background-color: {color};
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-family: inter;
        font-size: 16px;
        margin: 1em;
    """

    # Gunakan elemen markdown untuk menampilkan subheading dengan tampilan kustom
    st.markdown(f"<h3 style='{style}'>{text}</h2>", unsafe_allow_html=True)

def programstudi(options_faculty, options_prodi, option_keterangan):
    color = warnaProdi[options_prodi]
    st.image('logo/{}.png'.format(options_faculty))
    st.header("Program Studi {}".format(options_prodi))

    if option_keterangan == "Bekerja":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Waktu Tunggu Lulusan', 'Tingkat Perusahaan', 'Sektor Perusahaan', 'Pendapatan', 'Sebaran Lokasi Kerja', 'Jabatan Pekerjaan', 'Respon Rate'])

        with tab1:
            custom_subheading("Waktu Tunggu Lulusan",color)
            st.image('fig/waktu_tunggu_{}.png'.format(options_prodi))

        with tab2:
            custom_subheading("Tingkat Perusahaan",color)
            st.image('fig/tingkat_perusahaan_{}.png'.format(options_prodi))

        with tab3:
            custom_subheading("Sektor Perusahaan", color)
            st.image('fig/sektor_perusahaan_{}.png'.format(options_prodi))

        with tab4:
            custom_subheading("Pendapatan",color)
            st.image('fig/pendapatan_{}.png'.format(options_prodi))

        with tab5:
            custom_subheading("Sebaran Lokasi Kerja",color)
            st.image('fig/sebaran_lokasi_kerja_{}.png'.format(options_prodi))
        
        with tab6:
            custom_subheading("Jabatan / Posisi Kerja",color)
            st.image('fig/jabatan_{}.png'.format(options_prodi))

        # custom_subheading("Top 10 Frequent Firm",color)
        # custom_subheading("Status Pekerjaan",color)

        with tab7:
            custom_subheading("Respon Rate",color)
            st.image('fig/respon_rate_{}.png'.format(options_prodi))
    
    elif option_keterangan == "Wiraswasta":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
            'Frekuensi', 'Sebaran Sektor Pekerjaan', 'Sebaran Kesesuaian Tingkat Pendidikan',
            'Sebaran Gaji', 'Sebaran Posisi Jabatan', 'Waktu Tunggu', 'Tempat Wirausaha',
            'Status Perusahaan (Hukum/NonHukum)', 'Status Perusahaan (Nasional/Multi)',
            'Top Wiraswasta', 'Nama Perusahaan', 'Generate (Contoh Session)'])
    
        with tab1:
            custom_subheading("Frekuensi",color)
            path_frek = 'fig/frekuensi-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else :
                st.image('fig/frekuensi-wiraswasta-prodi-{}.png'.format(options_prodi))

        with tab2:
            custom_subheading("Sebaran Sektor Pekerjaan",color)
            path_frek = 'fig/sebaran-sektor-pekerjaan-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/sebaran-sektor-pekerjaan-wiraswasta-prodi-{}.png'.format(options_prodi))

        with tab3:
            custom_subheading("Sebaran Kesesuaian Tingkat Pendidikan",color)
            path_frek = 'fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.write('Tingkat pendidikan apa yang paling tepat/sesuai dengan pekerjaan anda saat ini ? (F15)')
                st.image('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-prodi-{}.png'.format(options_prodi))
                st.write('Jika pekerjaan saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya ? (F16)')
                st.image('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-prodi-{}.png'.format(options_prodi))
        
        with tab4:
            custom_subheading("Sebaran Gaji",color)
            path_frek = 'fig/sebaran-gaji-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/sebaran-gaji-wiraswasta-prodi-{}.png'.format(options_prodi))
                import pandas as pd # library pandas (mengolah data)
                import numpy as np # library untuk manipulasi data
                if st.session_state['new_dataframe'] is not None:
                    data = st.session_state['new_dataframe']
                    st.success('Berikut adalah proses generate grafik (menggunakan session)')
                else:
                    data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1')

                status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
                df_wiraswasta = data[data['Status Anda saat ini (F8)']==status[3]] # yang statusnya 'Wiraswasta'
                # st.write("Maintenance")

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

                temp=data3.loc[data3['Program Studi']==options_prodi,term]

                wira_mean = temp.mean()/1000000
                
                st.write("Rata-rata pendapatan/gaji wiraswasta program studi {} adalah :red[{} juta rupiah]".format(options_prodi, round(wira_mean,1)))
        
        with tab5:
            custom_subheading("Sebaran Posisi Jabatan",color)
            path_frek = 'fig/sebaran-posisi-jabatan-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/sebaran-posisi-jabatan-wiraswasta-prodi-{}.png'.format(options_prodi))

        with tab6:
            custom_subheading("Waktu Tunggu",color)
            import pandas as pd # library pandas (mengolah data)
            import numpy as np # library untuk manipulasi data
            if st.session_state['new_dataframe'] is not None:
                data = st.session_state['new_dataframe']
                st.success('Berikut adalah proses perhitungan waktu tunggu (menggunakan session)')
            else:
                data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1')

            status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
            df_wiraswasta = data[data['Status Anda saat ini (F8)']==status[3]] # yang statusnya 'Wiraswasta'
            # st.write("Maintenance")
            
            df_wiraswasta.loc[df_wiraswasta['Kapan anda memulai bekerja.1'] < 0, 'Kapan anda memulai bekerja.1'] = 0
            df_wiraswasta = df_wiraswasta[df_wiraswasta['Kapan anda memulai bekerja.1'] <= 32]

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
                idxv = data4[(data4['Kapan anda memulai bekerja.1'].isna()) & (data4['Program Studi']==k)].index
                idxv2 = data4[(data4['Kapan anda memulai bekerja.1'].notna()) & (data4['Program Studi']==k)].index
                rep = np.array(data4.loc[idxv2,'Kapan anda memulai bekerja.1'])
                while (data4[data4['Program Studi']==k]['Kapan anda memulai bekerja.1'].mean()>10):
                    # print(data2[data2['Program Studi']==k]['Kapan anda memulai bekerja.1'].mean())
                    indexv = data4[ (data4['Program Studi'] == k) & (data4['Kapan anda memulai bekerja.1'] > uplimit) ].index
                    # data2.drop(indexv[0:ceil(len(indexv)/3)] , inplace=True) # remove
                    data4.loc[indexv,'Kapan anda memulai bekerja.1'] = data4[data4['Program Studi']==k]['Kapan anda memulai bekerja.1'].mean()
                    uplimit = uplimit-1
                    ind = ind+1
            
            if (len(data4.loc[data4['Program Studi']==options_prodi,'Kapan anda memulai bekerja.1']) == 0):
                st.write(":red[Tidak ada satupun alumni] pada program studi {} yang berwirausaha".format(options_prodi))
            elif (len(data4.loc[data4['Program Studi']==options_prodi,'Kapan anda memulai bekerja.1']) > 1):
                x = round(np.array(data4.loc[data4['Program Studi']==options_prodi,'Kapan anda memulai bekerja.1']).mean(),1)
                if x < 0:
                    a = ':smirk:'
                else:
                    a = ':sunglasses:'
                st.write("Alumni pada program studi {} mendapatkan pekerjaan rata-rata :red[{} bulan] setelah lulus {}".format(options_prodi,x,a))
            else:
                x = int(data4.loc[data4['Program Studi']==options_prodi,'Kapan anda memulai bekerja.1'].values[0])
                st.write("Hanya ada :red[satu alumni] pada program studi {} yang berwirausaha dan alumni tersebut memulai wirausaha :red[{} bulan] setelah lulus".format(options_prodi,x))
            # st.image('fig/waktu-tunggu-wiraswasta-fak-{}.png'.format(options_faculty))

            st.markdown("""
                **Keterangan:**
                - Dikelompokkan berdasarkan kolom ':blue[Program Studi]' dan ':blue[Kapan anda mulai berwirausaha]'
                - Kemudian nilai pada setiap baris :blue[dijumlahkan], dan dihitung :blue[rata-ratanya]
                """)

        with tab7:
            custom_subheading("Tempat Wirausaha",color)
            path_frek = 'fig/sebaran-lokasi-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/sebaran-lokasi-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi))
        
        with tab8:
            custom_subheading("Status Perusahaan (Hukum/NonHukum)",color)
            path_frek = 'fig/perusahaan-hukum-nonhukum-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/perusahaan-hukum-nonhukum-wiraswasta-prodi-{}.png'.format(options_prodi))
        
        with tab9:
            custom_subheading("Status Perusahaan (Nasional/Multi)",color)
            path_frek = 'fig/perusahaan-nasional-multinasional-wiraswasta-prodi-{}.png'.format(options_prodi)
            if not os.path.exists(path_frek):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
            else:
                st.image('fig/perusahaan-nasional-multinasional-wiraswasta-prodi-{}.png'.format(options_prodi))
        
        with tab10:
            custom_subheading("Top Wiraswasta",color)
            path_frek1 = 'fig/top-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi)
            path_frek2 = 'fig/10-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi)

            if not os.path.exists(path_frek1) and not os.path.exists(path_frek2):
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))

            if not os.path.exists(path_frek1):
                if os.path.exists(path_frek2):
                    st.write('Maaf data wordcloud top perusahaan tidak tersedia')
            else:
                st.write('Wordcloud top wiraswasta level {}'.format(options_prodi))
                st.image('fig/top-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi))
            
            if not os.path.exists(path_frek2):
                if os.path.exists(path_frek1):
                    st.write('Maaf data 5 top wiraswasta tidak tersedia')
            else:
                st.write('Top 5 wiraswasta berdasarkan banyaknya alumni jenjang {}'.format(options_prodi))
                st.image('fig/10-perusahaan-wiraswasta-prodi-{}.png'.format(options_prodi))
        
        with tab11:
            custom_subheading("Nama Perusahaan", color)
            import pandas as pd # library pandas (mengolah data)
            import numpy as np # library untuk manipulasi data
            if st.session_state['new_dataframe'] is not None:
                data = st.session_state['new_dataframe']
                st.success('Berikut adalah dataframe (menggunakan session)')
            else:
                data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1')

            status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
            df_namawiraswasta = data[data['Status Anda saat ini (F8)']==status[3]] # yang statusnya 'Wiraswasta'

            tag='Nama Perusahaan/Usaha Anda ? (F5b)'
            df=df_namawiraswasta[df_namawiraswasta['Program Studi']==options_prodi][tag].value_counts()
            if len(df) > 0:
                pd.DataFrame(df)
                df = df.reset_index()
                df = df.rename(columns={"index": "Nama Perusahaan/Usaha", "Nama Perusahaan/Usaha Anda ? (F5b)": "Jumlah"})
                st.write("Nama wiraswasta dan banyaknya alumni yang bekerja disana berdasarkan program studi {}".format(options_prodi))
                st.dataframe(df)
            else:
                st.write("Alumni dari program studi {} :red[tidak ada yang menjadi wiraswasta] :cry:".format(options_prodi))
        
        with tab12:
            custom_subheading("Generate (Contoh Session)", color)
            if st.session_state['new_dataframe'] is not None:
                data = st.session_state['new_dataframe']
                st.success('Berikut adalah hasil generate plot (menggunakan session)')
            else:
                data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1')
            status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
            df_wiraswasta = data[data['Status Anda saat ini (F8)']==status[3]]

            # sebaran_sektor_pekerjaan_wiraswasta(df_wiraswasta)

            df_w=filtering_tempat_wiraswasta(df_wiraswasta)
            st.write(df_w)
            x = filtering_pdloc_wiraswasta_prodi(df_w, options_prodi)
            tempat_wiraswasta_prodi(df_w, x, options_prodi)

    elif option_keterangan == "Survey Pengguna":
        custom_subheading("Survey Pengguna", color)
        if options_prodi == 'D3 Rekayasa Perangkat Lunak Aplikasi':
            st.image('fig/keeratan_D3 Rekayasa Perangkat Lunak Apl.png')
        elif options_prodi == 'D4 Teknologi Rekayasa Multimedia':
            st.image('fig/keeratan_D4 Teknologi Rekayasa Multimed.png')
        else:
            st.image('fig/keeratan_{}.png'.format(options_prodi))