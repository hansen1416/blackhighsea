import { createApp } from 'vue'
// import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import router from './router'
import store from './store'
import i18n from './i18n'
// import VueSocketIO from 'vue-3-socket.io'
// import SocketIO from 'socket.io-client'
// import VueSocketIO from 'vue-socket.io'

// Vue.use(new VueSocketIO({
//     debug: true,
//     connection: 'http://metinseylan.com:1992',
//     vuex: {
//         store,
//         actionPrefix: 'SOCKET_',
//         mutationPrefix: 'SOCKET_'
//     },
//     options: { path: "/my-app/" } //Optional options
// }))

// new Vue({
//     router,
//     store,
//     render: h => h(App)
// }).$mount('#app')

// Vue.use(new VueSocketIO({
//     debug: true,
//     connection: 'http://metinseylan.com:1992',
//     vuex: {
//         store,
//         actionPrefix: 'SOCKET_',
//         mutationPrefix: 'SOCKET_'
//     },
//     options: { path: "/my-app/" } //Optional options
// }))
 
// new Vue({
//     router,
//     store,
//     render: h => h(App)
// }).$mount('#app')

createApp(App).use(i18n).use(store).use(router).mount('#app')

