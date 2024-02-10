import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_modal import Modal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from category_encoders import MEstimateEncoder

import tempfile
import os
import gdown
from gensim.models import KeyedVectors
import pickle

translate = {
    'product_link': 'Link mua hàng',
    'name': 'Tên sách',
    'detail_cate': 'Phân loại',
    'large_cate': 'Thể loại',
    'image': 'Hình ảnh',
    'price': 'Giá',
    'discount': 'Giảm giá',
    'sale_quantity': 'Đã bán',
    'rating_star': 'Đánh giá',
    'rating_quantity': 'Lượng đánh giá',
    'describe': 'Mô tả',
    'seller': 'Người bán',
    'seller_star': 'Đánh giá người bán',
    'seller_reviews_quantity': 'Lượng đánh giá người bán',
    'seller_follow': 'Lượng theo dõi người bán',
}

class func:
    def __init__(self):
        pass

    @staticmethod
    def cluster_data(data):
        def classify_cols(X):
            num_col = list(X.select_dtypes(['float64','int64','int32']).columns)
            highcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() > 10]
            lowcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() <= 10]
            return num_col, highcar_cat_col, lowcar_cat_col
        X = data.loc[:, ['price', 'detail_cate', 'large_cate']]
        y = data['sale_quantity']
        num_col, high_car_col, low_car_col = classify_cols(X)
        num_tfmer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scaling', StandardScaler())
        ])

        lowcar_tfmer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'most_frequent')),
            ('encode', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
        ])

        highcar_tfmer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'most_frequent')),
            ('encode', MEstimateEncoder()),
            ('scale', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers = [
            ('num', num_tfmer, num_col),
            ('high', highcar_tfmer, high_car_col),
            ('low', lowcar_tfmer, low_car_col)
        ])
        X_pp = preprocessor.fit_transform(X, y)
        km = KMeans(n_clusters = 7)
        data['clusters'] = km.fit_predict(X_pp)
        return data
    
    @staticmethod
    def get_vector(text, model):
        words = text.split(' ')
        vectors = [model[word] for word in words if word in model]
        return np.mean(vectors, axis = 0)
    
    @staticmethod
    def choose_similar_book(name, data, simi_df):
        try:
            cluster = int(name.loc[data['name'] == name]['clusters'])
        except:
            cluster = int(data.loc[data['name'] == name]['clusters'].iloc[0])
        cluster_matches = data.loc[data['clusters'] == cluster]['name'].tolist()
        bestbook = list(simi_df.loc[cluster_matches][name].sort_values(ascending = False)[1:20].index)
        return bestbook
    
    @staticmethod
    def chose_by_prompt(prompt, model, wv):
        prompt_vector = func.get_vector(prompt, model)
        books_simi = {}
        for b, v in wv.items():
            books_simi[b] = cosine_similarity([v], [prompt_vector])[0][0]
        sort_simi = sorted(books_simi.items(), key = lambda x: x[1], reverse = True)
        bestbook = [i[0] for i in sort_simi][1:20]
        return bestbook

class BookResource:
    def __init__(self, data_url, model_url):
        self.data = self.get_data(data_url)
        self.model = self.get_model(model_url)

    @staticmethod
    @st.cache_data
    def get_data(url):
        file_id = url.split('/')[-2]
        download_link = f"https://drive.google.com/uc?id={file_id}"
        data = pd.read_csv(download_link, index_col = 0)
        data.dropna(subset = ['price', 'detail_cate', 'large_cate'], inplace = True)
        data['describe'] = data['describe'].fillna('Không có mô tả')
        data = data.drop(['Phương thức giao hàng Seller Delivery',
                          'Địa chỉ tổ chức chịu trách nhiệm về hàng hóa',
                          'Tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa',
                          'Phiên bản', 'Dịch vụ nổi bật 2', 'Dịch vụ nổi bật 3'], axis = 1)
        return func.cluster_data(data)
    
    @staticmethod
    @st.cache_resource
    def get_model(model_url):
        fileid = model_url.split('/')[-2]
        url = f"https://drive.google.com/uc?id={fileid}"
        tempfile_name = 'wiki2.vn.vec'
        tempdir = tempfile.gettempdir()
        temp_path = os.path.join(tempdir, tempfile_name)
        gdown.download(url, temp_path, quiet = False)
        model = KeyedVectors.load_word2vec_format(temp_path)
        return model
     
    @staticmethod
    @st.cache_resource
    def get_wv(prepare = True, data = None, _model = None, 
               _wv_file_path = None, _simi_file_path = None):
        if prepare == False:
            wv = {}
            for n, v in zip(data['name'], data['describe']):
                wv[n] = func.get_vector(v, _model)
            wv_matrix = np.stack(list(wv.values()))
            simi = cosine_similarity(wv_matrix)
            vals = wv.keys()
            simi_df = pd.DataFrame(simi, columns = vals, index = vals)
            simi_df = simi_df.round(3)
        else:
            with open(_wv_file_path, 'rb') as f:
                wv = pickle.load(f)
            simi_df = pd.read_csv(_simi_file_path, index_col = 0)
        return wv, simi_df
    
class Style:
    def __init__(self, css_path):
        self.css = css_path
        with open(css_path, 'r', encoding = 'utf-8') as css:
            st.markdown(f'<style>{css.read()}<style>', unsafe_allow_html = True)
    
class Header:
    def __init__(self, header):
        self.header = header
        self.options = None

    def set_head_page(self):
        head_page = st.columns([0.3, 0.7])
        with head_page[0]:
            st.header(self.header)
        with head_page[1]:
            st.write('')
            with st.expander(label = 'THANH ĐIỀU HƯỚNG'):
                options = option_menu(
                    menu_title = 'MENU',
                    options = ['Hi', 'BOOK RECOMMENDER', 'BOOK MARKET', 'HOW THIS APP WORKS?'],
                    icons = ['robot','book','wrench'],
                    menu_icon = 'window-dock',
                    orientation = 'horizontal',
                    styles = {
                        'container': {'background-color': 'cornsilk', 'opacity': 0.8},
                        'nav-link': {'text-align': 'center',
                                    'font-family': "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif"},
                        "nav-link-selected": {"background-color": "green"}
                    })
        st.divider()
        self.options = options

class DefaultPage:
    def __init__(self, lottie_animation):
        self.animation = lottie_animation

    def show_page(self):
        dfpage = st.columns([0.3, 0.7])
        with dfpage[0]:
            try:
                st_lottie(self.animation)
            except:
                st.write('zzz')
        with dfpage[1]:
            st.write('')
            st.header('Xin chào bạn đến với Book Recommender. Bạn có thể chọn:')
            st.write('')
            st.subheader('- BOOK RECOMMENDER: Giúp bạn gợi ý sách dựa trên thông tin bạn chọn')
            st.write('')
            st.subheader('- BOOK MARKET: Xem tình hình thị trường sách ở Tiki như thế nào')
            st.write('')
            st.subheader('- HOW THIS APP WORKS: Cách mà chương trình hoạt động')

class filterBook:
    def __init__(self, data):
        self.data = data
        self.filter_data = None
        self.chose_book = None
        self.prompt = ''
        self.small_cate = None
        self.large_cate = None
        
    def show_page(self):
        tabs_col = st.columns([0.4, 0.6])
        with tabs_col[0]:
            chose_book = st.selectbox('Chọn sách', options = [None] + self.data['name'].unique().tolist())
            with st.expander('Chọn thể loại'):
                large_cate_list = ['Chọn tất cả'] + self.data['large_cate'].unique().tolist()
                large_cate = st.multiselect('Chọn nhóm lớn', options = large_cate_list, default = 'Chọn tất cả')
                small_cate_list = ['Chọn tất cả'] + self.data['detail_cate'].unique().tolist()
                small_cate = st.multiselect('Chọn nhóm nhỏ', options = small_cate_list, default = 'Chọn tất cả')
        if 'Chọn tất cả' in large_cate:
            if 'Chọn tất cả' in small_cate:
                data = self.data
            else:
                data = self.data[self.data['detail_cate'].isin(small_cate)]
        else:
            data = self.data[self.data['large_cate'].isin(large_cate)]
            if 'Chọn tất cả' in small_cate:
                data = data
            else:
                data = data[self.data['detail_cate'].isin(small_cate)]

        with tabs_col[1]:
            prompt = st.text_input("Mô tả về quyển sách bạn đang tìm",
                                placeholder = 'Tên sách, Nội dung ...')
            
        self.chose_book = chose_book
        self.prompt = prompt
        self.small_cate = small_cate
        self.large_cate = large_cate
        self.filter_data = data
        
class displayBook:
    def __init__(self, data, chose_book, prompt, wv, model, simi_df):
        self.data = data
        self.chose_book = chose_book
        self.prompt = prompt
        self.wv = wv
        self.model = model
        self.simi_df = simi_df
        self.display_data = self.show_book()
        self.check = False
        self.current_book = None
        self.display_data = self.display_data.rename(columns = translate)
    def show_book(self):
        if self.chose_book == None and self.prompt == '':
                df = self.data.sort_values(by = ['sale_quantity', 'rating_star', 'rating_quantity'], 
                                    ascending = False)
                df = df[:20]
        elif self.chose_book == None:
            best_book = func.chose_by_prompt(self.prompt, self.model, self.wv)
            df = self.data[self.data['name'].isin(best_book)].sort_values(by=['name'], 
                                                            key=lambda x: x.map({v: i for i, v in enumerate(best_book)}))
        else:
            best_book = func.choose_similar_book(self.chose_book, self.data, self.simi_df)
            df = self.data[self.data['name'].isin(best_book)].sort_values(by=['name'], 
                                                            key=lambda x: x.map({v: i for i, v in enumerate(best_book)}))
        return df

    def show_page(self):
        display = st.tabs(['Best match', 'Info', 'Describe'])
        with display[0]:
            i = 0
            for _ in range(5):
                with st.container():
                    book_row = st.columns(4)
                    for book in book_row:
                        with book:
                            try:
                                img_url = self.display_data['Hình ảnh'].tolist()[i]
                                st.image(img_url, use_column_width = True)
                                x = st.form(key = self.display_data['Tên sách'].tolist()[i])
                                with x:
                                    submit_button = st.form_submit_button(label = self.display_data['Tên sách'].tolist()[i])
                                if submit_button:
                                    self.check = True
                                    self.current_book = x._form_data[0]
                            except:
                                st.empty()
                            i += 1
        with display[1]:
            data_display1 = self.display_data.loc[:, [
                'Hình ảnh', 'Tên sách', 'Giá', 'Thể loại', 'Phân loại',
                'Đã bán', 'Dịch Giả', 'Kích thước', 'Số trang', 'Công ty phát hành',
                'Nhà xuất bản', 'Người bán'
            ]]
            st.dataframe(
                data_display1,
                column_config = {
                    'Hình ảnh': st.column_config.ImageColumn()
                },
                hide_index = True
            )
            self.df_display1 = data_display1
        with display[2]:
            data_display2 = self.display_data.loc[:,[
                'Hình ảnh', 'Tên sách', 'Mô tả'
            ]]
            st.dataframe(
                data_display2, 
                column_config = {
                    'Hình ảnh': st.column_config.ImageColumn(),
                    'Mô tả': st.column_config.TextColumn(width = 'large')
                    },
                hide_index = True
            )
            self.df_display2 = data_display2

class Popup:
    def __init__(self, check, current_book, data, max_width = 700):
        self.check = check
        self.current_book = current_book
        self.data = data
        self.max_width = max_width

    def run(self):
        if self.check == True:
            df = self.data[self.data['name'] == self.current_book]
            book_name = df['name'].tolist()[0]
            book_price = df['price'].tolist()[0]
            book_link = df['product_link'].tolist()[0]
            book_cate = df['large_cate'].tolist()[0]
            modal = Modal(title = book_name, key="Demo Key", max_width = self.max_width)
            with modal.container():
                st.header(body = '', divider = 'rainbow')
                show = st.columns(3)
                with show[0]:
                    st.info('Giá')
                    st.metric(label = '', value = f'{int(book_price):,}')
                with show[1]:
                    st.info('Thể loại')
                    st.write('')
                    st.subheader(f'{book_cate}')
                
                with show[2]:
                    st.info('Link')
                    st.write('')
                    st.markdown(f'<a href="{book_link}" target="_blank"> Click để mua sách </a>',
                                unsafe_allow_html = True)
                
                    
class bookMarket:
    class SideBar:
        def __init__(self, data):
            self.data = data
            self.large_cate = None
            self.small_cate = None

        def show(self):
            st.sidebar.header('PLEASE CHOOSE YOUR INFO YOU WANT TO SEE')
            large_cate = st.sidebar.multiselect(
                'Select Large Category:',
                options = ['SELECT ALL'] + list(self.data['Thể loại'].unique()),
                default = 'SELECT ALL'
            )
            small_cate = st.sidebar.multiselect(
                'Select Small Category',
                options = ['SELECT ALL'] + list(self.data[self.data['Thể loại'].isin(large_cate)]['Phân loại'].unique()),
                default = 'SELECT ALL'
            )

            if 'SELECT ALL' in large_cate:
                large_cate = self.data['Thể loại'].unique()

            if 'SELECT ALL' in small_cate:
                small_cate = self.data[self.data['Thể loại'].isin(large_cate)]['Phân loại'].unique()
            self.large_cate = large_cate
            self.small_cate = small_cate

    def __init__(self, data):
        data = data.rename(columns = translate)
        data['Số trang'] = pd.to_numeric(data['Số trang'], errors = 'coerce')
        data = data.dropna(subset = ['Đã bán', 'Lượng đánh giá', 'Số trang'])
        self.data = data
        self.sidebar = self.SideBar(self.data)
        self.filter_data = None
        self.check = False
        self.current_book = None

    def prepare_data(self):
        self.total_sales_quantity = self.filter_data['Đã bán'].sum()

        self.total_rating_quantity = self.filter_data['Lượng đánh giá'].sum()

        try:
            average_star = self.filter_data['Đánh giá'].sum() / len(self.filter_data['Đánh giá'])
        except:
            average_star = 0

        self.average_star = average_star

        self.best_seller = self.filter_data.sort_values(
            by = ['Đã bán', 'Lượng đánh giá'], 
            ascending = False
            )[:5]
        sales_by_covertype = self.filter_data.groupby(by = ['Loại bìa'])['Đã bán'].agg('sum').sort_values()
        sales_by_cate = self.filter_data.groupby(by = ['Thể loại', 'Phân loại'])['Đã bán'].agg('sum').sort_values()
        avg_pages_by_large = self.filter_data.groupby(by = ['Thể loại'])['Số trang'].agg('mean').sort_values()
        sales_by_company = self.filter_data.groupby(by = ['Công ty phát hành'])['Đã bán'].agg('sum').sort_values()
        sales_by_publisher = self.filter_data.groupby(by = ['Nhà xuất bản'])['Đã bán'].agg('sum').sort_values()
        ratequan_by_cate = self.filter_data.groupby(by = ['Thể loại'])['Lượng đánh giá'].agg('sum').sort_values()


        self.cover_type_px = px.pie(
            sales_by_covertype,
            names = sales_by_covertype.index,
            values = 'Đã bán',
            title = '<b> Sales Quantity by Cover Type <b>',
            color = 'Đã bán',
            template = 'simple_white',
        )

        self.cate_px = px.treemap(
            sales_by_cate,
            path = [sales_by_cate.index.get_level_values('Thể loại'),
                    sales_by_cate.index.get_level_values('Phân loại')],
            values = 'Đã bán',
            title = '<b> Sales Quantity by Category <b>',
            template = 'ggplot2'
        )

        self.avg_pages_large = px.bar(
            avg_pages_by_large,
            x = avg_pages_by_large.index,
            y = 'Số trang',
            title = '<b> Average Num of Pages by Large Category <b>',
            color = 'Số trang',
            color_continuous_scale = 'Blues',
            range_color = (0, 800),
        )

        self.company_px = px.bar(
            sales_by_company,
            x = 'Đã bán',
            y = sales_by_company.index,
            orientation = 'h',
            title = '<b> Sales Quantity by Publishing Company <b>',
            color = 'Đã bán',
            color_continuous_scale = 'fall'
        )

        self.publisher_px = px.bar(
            sales_by_publisher,
            x = 'Đã bán',
            y = sales_by_publisher.index,
            orientation = 'h',
            title = '<b> Sales Quantity by Publisher <b>',
            color = 'Đã bán',
            color_continuous_scale = 'deep'
        )

        self.rate_px = px.bar(
            ratequan_by_cate,
            x = 'Lượng đánh giá',
            y = ratequan_by_cate.index,
            orientation = 'h',
            title = '<b> Rate Quantity by Large Category <b>',
            color = 'Lượng đánh giá',
            color_continuous_scale = 'purpor'
        )

    def show_page(self):
        self.sidebar.show()
        self.filter_data = self.data[self.data['Thể loại'].isin(self.sidebar.large_cate)
                                     & self.data['Phân loại'].isin(self.sidebar.small_cate)]
        self.prepare_data()
        st.title('📗 📘BOOK MARKET DESCRIPTIVE ANALYSIS 📕 📙')
        info1, info2, info3 = st.columns(3, gap = 'large')
        with info1:
            st.info('Total Sales Quantity', icon = '📌')
            st.metric(label = 'Sum Sales Quantity', value = f'{self.total_sales_quantity:,.0f}')

        with info2:
            st.info('Total Rating Quantity', icon = '📌')
            st.metric(label = 'Sum Rating Quantity', value = f'{self.total_rating_quantity:,.0f}')

        with info3:
            st.info('Average Star Review', icon = '📌')
            st.metric(label = 'Average Star', value = f'{self.average_star:,.0f}')

        #BEST SELLER
        st.markdown('## Best Seller Book based on your filter:')
        
        bs = st.columns(5)
        try:
            for i, v in enumerate(bs):
                with v:
                    st.image(self.best_seller['Hình ảnh'].iloc[i])
                    x = st.form(key = self.best_seller['Tên sách'].iloc[i])
                    with x:
                        submit_button = st.form_submit_button(label = self.best_seller['Tên sách'].iloc[i])
                    if submit_button:
                        self.check = True
                        self.current_book = x._form_data[0]
        except:
            st.divider()

        cat, rate = st.columns([2, 1])
        with cat:
            st.plotly_chart(self.cate_px, use_container_width = True)
        with rate:
            st.plotly_chart(self.rate_px, use_container_width = True)


        com, pub = st.columns(2)
        with com:
            st.plotly_chart(self.company_px, use_container_width = True)
        with pub:
            st.plotly_chart(self.publisher_px, use_container_width = True)

        cov, pag = st.columns([1, 2])
        with cov:
            st.plotly_chart(self.cover_type_px, use_container_width = True)
        with pag:
            st.plotly_chart(self.avg_pages_large, use_container_width = True)
        
class explain:
    def __init__(self, data):
        self.data = data

    def show_page(self):
        st.title('HOW THIS APP WORKS?')
        st.divider()
        st.subheader('STEP1: WEB SCARPLING')
        st.text('')
        st.markdown('- The idea is access to Tiki and scarpling book info in: https://tiki.vn/sach-truyen-tieng-viet/')
        st.markdown('- Sample Data after processing:')
        st.write(self.data.drop(['clusters'], axis = 1).head())
        st.markdown('- Tool to use: Selenium with Threading')
        st.markdown('You can see detaily and use Web Scrapling in Tiki in my another project:\
                     <a href="https://github.com/HoangHao1009/hcrawler/"> hcrawler </a>', unsafe_allow_html = True)
        st.markdown('The text scarpling look like this:')
        with st.expander('Click to see scarpling code'):
            st.code(
                """
                from hcrawler import module

                #category link crawler'll take
                #example for book, it may be large category: dien-thoai-may-tinh-bang, thoi-trang-nu, ...
                #or small category: sach-van-hoc, sach-kinh-te,..
                root_link = 'https://tiki.vn/sach-truyen-tieng-viet/c316' 
                #Numbers of chrome drivers will open for crawl
                n_browers = 5
                #CSS SELECTOR for elements (those behind are collected in Feb-4-2024)
                prod_link_elem = '.style__ProductLink-sc-1axza32-2.ezgRFw.product-item'
                category_bar_elem = '.breadcrumb'
                image_elem = '.image-frame'
                price_elem = '.product-price__current-price'
                discount_elem = '.product-price__discount-rate'
                sales_quantity_elem = '.styles__StyledQuantitySold-sc-1swui9f-3.bExXAB'
                rating_elem = '.styles__StyledReview-sc-1swui9f-1.dXPbue'
                info_elem = '.WidgetTitle__WidgetContainerStyled-sc-1ikmn8z-0.iHMNqO'
                detail_info_elem = '.WidgetTitle__WidgetContentStyled-sc-1ikmn8z-2.jMQTPW'
                describe_elem = '.style__Wrapper-sc-13sel60-0.dGqjau.content'
                extend_page_elem = '.btn-more'
                title_elem = '.WidgetTitle__WidgetTitleStyled-sc-1ikmn8z-1.eaKcuo'
                #sub_link_elem will be used for crawl detail category in root_link you put
                sub_link_elem = '.styles__TreeItemStyled-sc-1uq9a9i-2.ThXqv a'

                #you can put extra preventive CSS elements if prod_link_elem or sub_link_elem isn't valid
                preventive_prod_link_elem = '.style__ProductLink-sc-139nb47-2.cKoUly.product-item'
                preventive_sub_link_elem = '.item.item--category'

                crawler = module.TikiCrawler(root_link, n_browers, 
                             prod_link_elem, category_bar_elem, image_elem, 
                             price_elem, discount_elem,
                             sales_quantity_elem, rating_elem,
                             info_elem, detail_info_elem,
                             describe_elem,
                             extend_page_elem,
                             title_elem, preventive_prod_link_elem)

                crawler.crawl_multipage(50)
                #save data you've crawled
                crawler.save('Tikibook50crawler.pickle')
                """
            ,language = 'python')

        st.subheader('STEP2: TEXT PROCESSING USING DEEP LEARNING')
        st.text('')
        st.markdown('- In this I use Natural Language Processing to transform text data')
        st.markdown('- Tool to use: Gensim FastText for text vectorizing')
        st.markdown('Use FastText model build for Vietnamese: cc.vi.300.bin')
        with st.expander('Click to see sample text processing code:'):
            st.code(
                """
                @staticmethod
                @st.cache_data
                def get_data(url):
                    file_id = url.split('/')[-2]
                    download_link = f"https://drive.google.com/uc?id={file_id}"
                    data = pd.read_csv(download_link, index_col = 0)
                    data.dropna(subset = ['price', 'detail_cate', 'large_cate'], inplace = True)
                    data['describe'] = data['describe'].fillna('Không có mô tả')
                    data = data.drop(['Phương thức giao hàng Seller Delivery',
                                    'Địa chỉ tổ chức chịu trách nhiệm về hàng hóa',
                                    'Tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa',
                                    'Phiên bản', 'Dịch vụ nổi bật 2', 'Dịch vụ nổi bật 3'], axis = 1)
                    return func.cluster_data(data)
                
                @staticmethod
                @st.cache_resource
                def get_model(temp_path):
                    model = KeyedVectors.load_word2vec_format(temp_path)
                    return model
                
                @staticmethod
                @st.cache_resource
                def get_wv(prepare = True, data = None, _model = None, 
                        _wv_file_path = None, _simi_file_path = None):
                    if prepare == False:
                        wv = {}
                        for n, v in zip(data['name'], data['describe']):
                            wv[n] = func.get_vector(v, _model)
                        wv_matrix = np.stack(list(wv.values()))
                        simi = cosine_similarity(wv_matrix)
                        vals = wv.keys()
                        simi_df = pd.DataFrame(simi, columns = vals, index = vals)
                        simi_df = simi_df.round(3)
                    else:
                        with open(_wv_file_path, 'rb') as f:
                            wv = pickle.load(f)
                        simi_df = pd.read_csv(_simi_file_path)
                    return wv, simi_df
                """
            ,language = 'python')

        st.subheader('STEP3: BOOK CHOSING ALGORITHMS USING MACHINE LEANRING')
        st.text('')
        st.markdown('- First I clustering book by its detail category, large category and price')
        with st.expander('Click to see sample clustering code'):
            st.code(
                """
            def cluster_data(data):
                def classify_cols(X):
                    num_col = list(X.select_dtypes(['float64','int64','int32']).columns)
                    highcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() > 10]
                    lowcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() <= 10]
                    return num_col, highcar_cat_col, lowcar_cat_col
                X = data.loc[:, ['price', 'detail_cate', 'large_cate']]
                y = data['sale_quantity']
                num_col, high_car_col, low_car_col = classify_cols(X)
                num_tfmer = Pipeline(steps = [
                    ('impute', SimpleImputer(strategy = 'median')),
                    ('scaling', StandardScaler())
                ])

                lowcar_tfmer = Pipeline(steps = [
                    ('impute', SimpleImputer(strategy = 'most_frequent')),
                    ('encode', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
                ])

                highcar_tfmer = Pipeline(steps = [
                    ('impute', SimpleImputer(strategy = 'most_frequent')),
                    ('encode', MEstimateEncoder()),
                    ('scale', StandardScaler())
                ])

                preprocessor = ColumnTransformer(transformers = [
                    ('num', num_tfmer, num_col),
                    ('high', highcar_tfmer, high_car_col),
                    ('low', lowcar_tfmer, low_car_col)
                ])
                X_pp = preprocessor.fit_transform(X, y)
                km = KMeans(n_clusters = 7)
                data['clusters'] = km.fit_predict(X_pp)
                return data
                """
            , language = 'python')

        st.markdown('- And then i make 2 way to chosing book: by specify one or describe a book')
        with st.expander('Click to see sample chosing book code'):
            st.code(
                """
    @staticmethod
    def choose_similar_book(name, data, simi_df):
        try:
            cluster = int(name.loc[data['name'] == name]['clusters'])
        except:
            cluster = int(data.loc[data['name'] == name]['clusters'].iloc[0])
        cluster_matches = data.loc[data['clusters'] == cluster]['name'].tolist()
        bestbook = list(simi_df.loc[cluster_matches][name].sort_values(ascending = False)[1:20].index)
        return bestbook
    
    @staticmethod
    def chose_by_prompt(prompt, model, wv):
        prompt_vector = func.get_vector(prompt, model)
        books_simi = {}
        for b, v in wv.items():
            books_simi[b] = cosine_similarity([v], [prompt_vector])[0][0]
        sort_simi = sorted(books_simi.items(), key = lambda x: x[1], reverse = True)
        bestbook = [i[0] for i in sort_simi][1:20]
        return bestbook
                """
            , language = 'python')

        st.markdown('That is what I do for recommending book. Hope you enjoy it.')

class Footer:
    def __init__(self, avatar_img,
                 linkedin_link, github_link, facebook_link):
        self.avatar_img = avatar_img
        self.linkedin_link = linkedin_link
        self.github_link = github_link
        self.facebook_link = facebook_link

    def show_page(self):
        for _ in range(10):
            st.write('')
        st.divider()
        footer = st.container()
        with footer:
            e1, e2, e3, e4 = st.columns([0.5, 0.5, 2, 2])
            with e1:
                img_url = self.avatar_img
                st.image(img_url, use_column_width = True)
            with e2:
                st.empty()
            with e3:
                st.markdown('👨‍💻 Hoàng Hảo')
                st.markdown('🏠 Ho Chi Minh City')
                st.markdown('📞 Phone: 0866 131 594')
                st.markdown('✉️ hahoanghao1009@gmail.com')
            with e4:
                i1, i2, i3 = st.columns(3)
                with i1:
                    image_url = 'https://cdn-icons-png.flaticon.com/256/174/174857.png'
                    linkedin_url = self.linkedin_link

                    clickable_image_html = f"""
                        <a href="{linkedin_url}" target="_blank">
                            <img src="{image_url}" alt="Clickable Image" width="50">
                        </a>
                    """
                    st.markdown(clickable_image_html, unsafe_allow_html=True)

                with i2:
                    image_url = 'https://cdn-icons-png.flaticon.com/512/25/25231.png'
                    git_url = self.github_link

                    clickable_image_html = f"""
                        <a href="{git_url}" target="_blank">
                            <img src="{image_url}" alt="Clickable Image" width="50">
                        </a>
                    """
                    st.markdown(clickable_image_html, unsafe_allow_html=True)
                with i3:
                    image_url = 'https://cdn-icons-png.flaticon.com/512/3536/3536394.png'
                    fb_url = self.facebook_link

                    clickable_image_html = f"""
                        <a href="{fb_url}" target="_blank">
                            <img src="{image_url}" alt="Clickable Image" width="50">
                        </a>
                    """
                    st.markdown(clickable_image_html, unsafe_allow_html=True)
        st.divider()

st.set_page_config(
    page_title = 'BOOK RECOMMEDER',
    layout = 'wide'
)

bookresource = BookResource(
    'https://drive.google.com/file/d/1fc9JI6nTzOOa1XNwktONFcSARqL71rOH/view?usp=sharing',
    'https://drive.google.com/file/d/1gOSC8hEnmsXQ6A0Gp-W-mG-JK39Mpguz/view?usp=sharing',
)
bookresource.wv, bookresource.simi_df = BookResource.get_wv(
    prepare = True,
    _wv_file_path = 'C:/Users/admin/Documents/Me/book_recommender/data/wv.pickle',
    _simi_file_path = 'C:/Users/admin/Documents/Me/book_recommender/data/simi_df.csv'
)


page_config = Style(
    'C:/Users/admin/Documents/Me/book_recommender/webapp_2/pagestyle.css',
)


header = Header('BOOK RECOMMENDER')
header.set_head_page()

check = False
current_book = None


if header.options == 'Hi':
    defaultpage = DefaultPage(
        'https://lottie.host/84d5a24a-eec9-482f-a7c7-0928268213a2/md07jHqM0X.json'
    )
    defaultpage.show_page()

elif header.options == 'BOOK RECOMMENDER':
    filter = filterBook(bookresource.data)
    filter.show_page()

    display = displayBook(
        filter.filter_data,
        filter.chose_book,
        filter.prompt,
        bookresource.wv,
        bookresource.model,
        bookresource.simi_df
    )
    display.show_page()
    check = display.check
    current_book = display.current_book

elif header.options == 'BOOK MARKET':
    bookmarket = bookMarket(bookresource.data)
    bookmarket.show_page()
    check = bookmarket.check
    current_book = bookmarket.current_book

elif header.options == 'HOW THIS APP WORKS?':
    explainer = explain(bookresource.data)
    explainer.show_page()


popup = Popup(
    check,
    current_book,
    bookresource.data
)
popup.run()

footer = Footer(
    'https://scontent.fhan3-4.fna.fbcdn.net/v/t39.30808-6/300214291_923781189013440_1485100982149543062_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=efb6e6&_nc_eui2=AeHP3OcZTMPNG0ZTR2SiwpURJWpGnT32TTIlakadPfZNMgWTvhMvJdKH8CBvQJ5VgiRu1qOUf6Ym7lhadFEwEtom&_nc_ohc=9qJ91-N9oKkAX-3EoqY&_nc_ht=scontent.fhan3-4.fna&oh=00_AfAs4ZdQnAAaP2l4Rxfl8I4h2EmWGSVL27XNpdIlhKUp4g&oe=65C7C2C2',
    'https://www.linkedin.com/in/hahoanghao1009/',
    'https://github.com/HoangHao1009/',
    'https://www.facebook.com/hoanghao1009/'
)
footer.show_page()