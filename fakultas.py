import streamlit as st
from streamlit_option_menu import option_menu

warnaFakultas = {"FIF": "#B28D35",
            "FRI": "#0B6623",
            "FTE": "#000080" ,
            "FEB":"#5959FF",
            "FKB": "#8510BC",
            "FIK": "#FF7F00",
            "FIT": "#3EB049"}

def custom_subheading(text,color):
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


def fakultas(options_faculty, option_keterangan):
    color = warnaFakultas[options_faculty]
    st.image('logo/fif.png'.format(options_faculty))
    if option_keterangan == 'Bekerja':
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Waktu Tunggu Lulusan', 'Tingkat Perusahaan', 'Pendapatan', 'Sebaran Lokasi Kerja', 'Jabatan Pekerjaan', 'Respon Rate'])

        with tab1:
            custom_subheading("Waktu Tunggu Lulusan",color)
            st.image('fig/waktu_tunggu_{}.png'.format(options_faculty))

        with tab2:
            custom_subheading("Tingkat Perusahaan",color)
            st.image('fig/tingkat_perusahaan_{}.png'.format(options_faculty))

        # custom_subheading("Sektor Perusahaan")
        with tab3:
            custom_subheading("Pendapatan",color)
            st.image('fig/pendapatan_{}.png'.format(options_faculty))

        with tab4:
            custom_subheading("Sebaran Lokasi Kerja",color)
            st.image('fig/sebaran_lokasi_kerja_{}.png'.format(options_faculty))
        
        with tab5:
            custom_subheading("Jabatan / Posisi Kerja",color)
            st.image('fig/jabatan_{}.png'.format(options_faculty))

        # custom_subheading("Top 10 Frequent Firm",color)
        # custom_subheading("Status Pekerjaan",color)

        with tab6:
            custom_subheading("Respon Rate",color)
            st.image('fig/respon_rate_{}.png'.format(options_faculty))
    
    elif option_keterangan == 'Wiraswasta':
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            'Frekuensi', 'Sebaran Sektor Pekerjaan', 'Sebaran Kesesuaian Tingkat Pendidikan',
            'Sebaran Gaji', 'Sebaran Posisi Jabatan', 'Waktu Tunggu', 'Tempat Wirausaha',
            'Status Perusahaan (Hukum/NonHukum)', 'Status Perusahaan (Nasional/Multi)',
            'Top Wiraswasta', 'Nama Perusahaan'])
        
        with tab1:
            custom_subheading("Frekuensi",color)
            st.image('fig/populasi-wiraswasta-fak-{}.png'.format(options_faculty))
            st.write('Frekuensi level fakultas {}'.format(options_faculty))
            st.image('fig/frekuensi-wiraswasta-fak-{}.png'.format(options_faculty))

        with tab2:
            custom_subheading("Sebaran Sektor Pekerjaan",color)
            st.image('fig/sebaran-sektor-pekerjaan-wiraswasta-fak-{}.png'.format(options_faculty))

        with tab3:
            custom_subheading("Sebaran Kesesuaian Tingkat Pendidikan",color)
            st.write('Tingkat pendidikan apa yang paling tepat/sesuai dengan wirausaha anda?')
            st.image('fig/sebaran-tingkat-kesesuaian-pendidikan-wiraswasta-fak-{}.png'.format(options_faculty))
            st.write('Jika wirausaha saat ini tidak sesuai dengan pendidikan anda, mengapa anda mengambilnya?')
            st.image('fig/sebaran-pertanyaan-kesesuaian-pendidikan-wiraswasta-fak-{}.png'.format(options_faculty))
        
        with tab4:
            custom_subheading("Sebaran Gaji",color)
            st.image('fig/sebaran-gaji-wiraswasta-fak-{}.png'.format(options_faculty))
            st.write('Sebaran Pendapatan Fakultas {}'.format(options_faculty))
            st.image('fig/gaji-wiraswasta-fak-{}.png'.format(options_faculty))
        
        with tab5:
            custom_subheading("Sebaran Posisi Jabatan",color)
            st.image('fig/sebaran-posisi-jabatan-wiraswasta-fak-{}.png'.format(options_faculty))

        with tab6:
            custom_subheading("Waktu Tunggu",color)
            st.image('fig/waktu-tunggu-wiraswasta-fak-{}.png'.format(options_faculty))

        with tab7:
            custom_subheading("Tempat Wirausaha",color)
            st.image('fig/sebaran-lokasi-perusahaan-wiraswasta-fak-{}.png'.format(options_faculty))
        
        with tab8:
            custom_subheading("Status Perusahaan (Hukum/NonHukum)",color)
            st.image('fig/perusahaan-hukum-nonhukum-wiraswasta-fak-{}.png'.format(options_faculty))
        
        with tab9:
            custom_subheading("Status Perusahaan (Nasional/Multi)",color)
            st.image('fig/perusahaan-nasional-multinasional-wiraswasta-fak-{}.png'.format(options_faculty))
        
        with tab10:
            custom_subheading("Top Wiraswasta",color)
            st.write('Wordcloud top perusahaan wiraswasta level {}'.format(options_faculty))
            st.image('fig/top-perusahaan-wiraswasta-fak-{}.png'.format(options_faculty))
            st.write('Top 5 wiraswasta berdasarkan banyaknya alumni jenjang {}'.format(options_faculty))
            st.image('fig/10-perusahaan-wiraswasta-fak-{}.png'.format(options_faculty))

        with tab11:
            custom_subheading("Nama Perusahaan", color)
            import pandas as pd # library pandas (mengolah data)
            import numpy as np # library untuk manipulasi data
            data = pd.read_excel('data/Tracer Final 2022.xlsx', sheet_name='Sheet1') # membaca dataset Tracer Final 2022.xlsx sheet 1

            status = list(data['Status Anda saat ini (F8)'].unique()) # mencari nilai
            df_namawiraswasta = data[data['Status Anda saat ini (F8)']==status[3]] # yang statusnya 'Wiraswasta'

            tag='Nama Perusahaan/Usaha Anda ? (F5b)'
            df=df_namawiraswasta[df_namawiraswasta['Fakultas']==options_faculty][tag].value_counts()
            pd.DataFrame(df)
            df = df.reset_index()
            df = df.rename(columns={"index": "Nama Perusahaan/Usaha", "Nama Perusahaan/Usaha Anda ? (F5b)": "Jumlah"})
            st.write("Nama wiraswasta dan banyaknya alumni yang bekerja disana berdasarkan fakultas {}".format(options_faculty))
            st.dataframe(df)

    elif option_keterangan == "Survey Pengguna":
        custom_subheading("Survey Pengguna", color)
        st.image('fig/keeratan_{}.png'.format(options_faculty))
