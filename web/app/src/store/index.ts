import { createStore } from 'vuex';

export default createStore({
  state: {
    stocknames: {}
  },
  mutations: {
    setStocknames(state: any, data: Record<string, string>) {
        state.stocknames = data;
    },
  },
  actions: {
  },
  modules: {

  }
})
