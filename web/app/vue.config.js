module.exports = {
  pluginOptions: {
    i18n: {
      locale: 'en',
      fallbackLocale: 'en',
      localeDir: 'locales',
      enableLegacy: false,
      runtimeOnly: false,
      compositionOnly: false,
      fullInstall: true
    }
  },
  devServer: {
    hot: false,
    liveReload: false
  }
//   devServer: {
//     proxy: {
//         '/socket.io': {
//             target: 'http://localhost:4601',
//             ws: true,
//             changeOrigin: true,
//         }
//     }
// }
}
