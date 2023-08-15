def wordclouddraw(data):

    from wordcloud import WordCloud
    wordcloud = WordCloud(max_words=1500, width=600, background_color='black').generate(" ".join(data['Review']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Most comnmon words")
    plt.axis("off")
    plt.show()