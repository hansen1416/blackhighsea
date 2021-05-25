import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/Home.vue')
  },
  {
    path: '/stylize',
    name: 'stylize',
    component: () => import('@/views/Stylize.vue')
  },
  {
    path: '/stocks',
    name: 'stock_list',
    component: () => import('@/views/StockList.vue')
  },
  {
    path: '/pred/hl/:stock_code',
    name: 'pred_hl',
    component: () => import('@/views/LineChart.vue')
  },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
