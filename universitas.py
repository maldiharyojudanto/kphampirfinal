import streamlit as st
from streamlit_option_menu import option_menu

def custom_subheading(text):
    # Definisikan CSS inline untuk menyesuaikan tampilan teks
    style = f"""
        color: black;
        background-color: #D9D9D9;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-family: inter;
        font-size: 16px;
        margin: 1em;
    """

    # Gunakan elemen markdown untuk menampilkan subheading dengan tampilan kustom
    st.markdown(f"<h3 style='{style}'>{text}</h2>", unsafe_allow_html=True)

def universitas(keterangan):
    st.header("Universitas")
    if keterangan == "Bekerja":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(['Waktu Tunggu Lulusan', 'Tingkat Perusahaan', 'Sektor Perusahaan', 'Pendapatan', 'Sebaran Lokasi Kerja', 'Jabatan Pekerjaan', 'Top 10 Frequent Firm', 'Status Pekerjaan', 'Respon Rate'])
        
        with tab1:
            custom_subheading("Waktu Tunggu Lulusan")
            st.image('fig/wkt_tunggu_univ.png')
            st.write('Waktu Tunggu per Program Studi')
            st.image('fig/waktu-tunggu-semua-prodi.png')

        with tab2:
            custom_subheading("Tingkat Perusahaan")
            st.image('fig/tingkat_perusahaan_univ.png')

        with tab3:
            custom_subheading("Sektor Perusahaan")
            st.image('fig/sektor_perusahaan_univ.png')
        
        with tab4:
            custom_subheading("Pendapatan")
            st.image('fig/pendapatan_univ.png')
            st.write('Sebaran Pendapatan per Program Studi')
            st.image('fig/gaji-semua-prodi.png')

        with tab5:
            custom_subheading("Sebaran Lokasi Kerja")
            st.image('fig/sebaran_lokasi_kerja_univ.png')

        with tab6:
            custom_subheading("Jabatan / Posisi Kerja")
            st.image('fig/jabatan_univ.png')

        with tab7:
            custom_subheading("Top 10 Frequent Firm")
            st.image('fig/top10perusahaan_univ.png')

        with tab8:
            custom_subheading("Status Pekerjaan")
            st.image('fig/status_pekerjaan_univ.png')

        with tab9:
            custom_subheading("Respon Rate")
            st.image('fig/respon_rate_univ.png')
            
    elif keterangan == "Melanjutkan Pendidikan":
        tab1, tab2, tab3, tab4 = st.tabs(['Frekuensi', 'Sumber Biaya', 'Sebaran Universitas', 'Sebaran Program Studi'])

        with tab1:
            custom_subheading("Frekuensi")
            st.image('fig/populasi-melanjutkan-pendidikan-univ.png')
            st.write('Frekuensi level universitas dikelompokkan dengan fakultas')
            st.image('fig/frekuensi-melanjutkan-pendidikan-fakuniv.png')
            st.write('Frekuensi level universitas dikelompokkan dengan prodi')
            st.image('fig/frekuensi-melanjutkan-pendidikan-prodiuniv.png')

        with tab2:
            custom_subheading("Sumber Biaya")
            st.image('fig/sumber-biaya-melanjutkan-univ.png')

        with tab3:
            custom_subheading("Sebaran Universitas")
            st.write('Wordcloud sebaran universitas tujuan alumni yang melanjutkan pendidikan')
            st.image('fig/sebaran-melanjutkan-pendidikan-univ.png')
            st.write('Top universitas berdasarkan banyaknya alumni yang melanjutkan pendidikan')
            st.image('fig/top-universitas-melanjutkan-univ.png')

        with tab4:
            custom_subheading("Sebaran Program Studi")
            st.write('Wordcloud sebaran program studi tujuan alumni yang melanjutkan pendidikan')
            st.image('fig/sebaran-melanjutkan-pendidikan-prodi.png')
            st.write('Top program studi berdasarkan banyaknya alumni yang melanjutkan pendidikan')
            st.image('fig/top-prodi-melanjutkan-prodi.png')

    elif keterangan == "Tidak Bekerja":
        tab1, tab2= st.tabs(['Frekuensi', 'Level Motivasi'])

        with tab1:
            custom_subheading("Frekuensi")
            st.image('fig/populasi-tidak-bekerja-univ.png')
            st.write('Frekuensi level universitas dikelompokkan dengan fakultas')
            st.image('fig/frekuensi-tidak-bekerja-fakuniv.png')
            st.write('Frekuensi level universitas dikelompokkan dengan prodi')
            st.image('fig/frekuensi-tidak-bekerja-prodiuniv.png')

        with tab2:
            custom_subheading("Level Motivasi")
            st.latex(r'''
                        Level\ Motivasi = round\left(\frac{1}{4}\sum_{i=1}^{4}A_i\right)
                        ''')
            st.markdown("""
                        - $A_1$: Berapa perusahaan/instansi/institusi yang sudah anda lamar (lewat surat atau E-mail)? (F6)
                        - $A_2$: Berapa banyak perusahaan/instansi/institusi yang merespon lamaran anda ? (F7)
                        - $A_3$: Berapa banyak perusahaan/instansi/institusi yang mengundang anda untuk wawancara ? (F7a)
                        - $A_4$: Apakah anda aktif mencari pekerjaan dalam 4 minggu terakhir ? (F10)
                        """)
            st.image('fig/level-motivasi-tidak-bekerja-univ.png')
        
    elif keterangan == "Wiraswasta":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            'Frekuensi', 'Sebaran Sektor Pekerjaan', 'Sebaran Kesesuaian Tingkat Pendidikan',
            'Sebaran Gaji', 'Sebaran Posisi Jabatan', 'Waktu Tunggu', 'Tempat Wirausaha',
            'Status Perusahaan (Hukum/NonHukum)', 'Status Perusahaan (Nasional/Multi)',
            'Top Wiraswasta', 'Nama Perusahaan'])
        
        with tab1:
            custom_subheading("Frekuensi")
            st.image('fig/populasi-wiraswasta-univ.png')
            st.write('Frekuensi level universitas dikelompokkan dengan fakultas')
            st.image('fig/frekuensi-wiraswasta-fakuniv.png')
            st.write('Frekuensi level universitas dikelompokkan dengan prodi')
            st.image('fig/frekuensi-wiraswasta-prodiuniv.png')

        with tab2:
            custom_subheading("Sebaran Sektor Pekerjaan")
            st.image('fig/sebaran-sektor-pekerjaan-wiraswasta-univ.png')

        with tab3:
            custom_subheading("Sebaran Kesesuaian Tingkat Pendidikan")
            st.write('Tingkat pendidikan apa yang paling tepat/sesuai dengan wirausaha anda?')
            st.image('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-univ.png')
            st.write('Jika wirausaha saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya?')
            st.image('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-univ.png')
        
        with tab4:
            custom_subheading("Sebaran Gaji")
            st.image('fig/sebaran-gaji-wiraswasta-univ.png')
            st.write('Sebaran Pendapatan per Program Studi')
            st.image('fig/gaji-wiraswasta-prodi.png')
        
        with tab5:
            custom_subheading("Sebaran Posisi Jabatan")
            st.image('fig/sebaran-posisi-jabatan-wiraswasta-univ.png')

        with tab6:
            custom_subheading("Waktu Tunggu")
            st.image('fig/waktu-tunggu-wiraswasta-univ.png')

        with tab7:
            custom_subheading("Tempat Wirausaha")
            st.image('fig/sebaran-lokasi-perusahaan-wiraswasta-univ.png')
        
        with tab8:
            custom_subheading("Status Perusahaan (Hukum/NonHukum)")
            st.image('fig/perusahaan-hukum-nonhukum-wiraswasta-univ.png')
        
        with tab9:
            custom_subheading("Status Perusahaan (Nasional/Multi)")
            st.image('fig/perusahaan-nasional-multinasional-wiraswasta-univ.png')
        
        with tab10:
            custom_subheading("Top Wiraswasta")
            st.write('Wordcloud top wiraswasta level universitas')
            st.image('fig/top-perusahaan-wiraswasta-univ.png')
            st.write('Top 5 wiraswasta berdasarkan banyaknya alumni level universitas')
            st.image('fig/10-perusahaan-wiraswasta-univ.png')
        
        with tab11:
            custom_subheading("Nama Perusahaan")
            import pandas as pd # library pandas (mengolah data)
            import numpy as np # library untuk manipulasi data
            data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1') # membaca dataset Tracer Final 2022.xlsx sheet 1

            status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
            df_namawiraswasta = data[data['Status Anda saat ini (F8)']==status[3]] # yang statusnya 'Wiraswasta'

            tag='Nama Perusahaan/Usaha Anda ? (F5b)'
            df=df_namawiraswasta[tag].value_counts()
            pd.DataFrame(df)
            df = df.reset_index()
            df = df.rename(columns={"index": "Nama Perusahaan/Usaha", "Nama Perusahaan/Usaha Anda ? (F5b)": "Jumlah"})
            st.write("Nama perusahaan wiraswasta dan banyaknya alumni yang bekerja level universitas")
            st.dataframe(df)

    elif keterangan == "Survey Pengguna":
        custom_subheading("Survey Pengguna")
        st.image('fig/keeratan_univ.png')
        