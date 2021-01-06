<template>
  <section>
    <div class='home'>
      <router-link
        :to="{name: 'home'}"
      >
        <span>home</span>
      </router-link>
    </div>
    <div class="box">
      <canvas ref="chart"></canvas>
    </div>
  </section>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import axios, {AxiosResponse} from "axios";
import Chart from 'chart.js';

interface Data{
  date: string[];
  real_close: number[];
  pred_close: number[];
  next_close: number[];
}

export default defineComponent({
  props: {
    
  },
  data() {
    return {

    }
  },
  computed: {
    ctx(): CanvasRenderingContext2D {
      return (this.$refs.chart as HTMLCanvasElement).getContext('2d') as CanvasRenderingContext2D;
    },
    stockCode(): string {
      return this.$route.params.stock_code as string;
    },
    stockName(): string {
      if (this.$store.state.stocknames && this.$store.state.stocknames[this.stockCode]) {
        return this.$store.state.stocknames[this.stockCode];
      } else {
        return '';
      }
    }
  },
  methods: {
    drawChart(data: Data, y_max: number, y_min: number) {
      new Chart(this.ctx, {
            
              type: 'line',
              data: {
                  labels: data.date,
                  datasets: [
                  {
                    data: data.real_close,
                    borderColor: "rgb(233, 60, 88)",
                    fill: false,
                    pointRadius: 0,
                  }, 
                  {
                    data: data.pred_close,
                    borderColor: "rgba(233, 60, 88, .6)",
                    fill: false,
                    pointRadius: 0,
                  }, 
                  {
                    data: data.next_close,
                    borderColor: "rgb(233, 60, 88)",
                    fill: false,
                    pointRadius: 5,
                    pointBackgroundColor: "rgb(233, 60, 88)",
                  },
                  ]
              },
              options: {
                  title: {
                      display: true,
                      text: `${this.stockCode} ${this.stockName} Price prediction`,
                  },
                  scales: {
                    yAxes: [{
                          display: true,
                          ticks: {
                              suggestedMin: y_min,
                              min: y_min,
                              max: y_max,
                          }
                    }]
                  },
                  legend: {
                    display: false,
                  }
                  // layout: {
                  //   padding: {
                  //     left: 0,
                  //     right: 0,
                  //     top: 0,
                  //     bottom: 0,
                  //   },
                  // },
              }
          });
    }
  },
  mounted() {

    axios.get(`${process.env.VUE_APP_PY_API}/prediction/${this.stockCode}`)
    .then((response: AxiosResponse) => {
      if(response.status == 200) {

        const data: Data = response.data;
        
        // pop empty data
        data.date.push('', '')
        
        data.real_close.pop()

        const data_len = data.real_close.length;

        const limit = 50;

        data.date = data.date.slice(data_len - limit);
        data.real_close = data.real_close.slice(data_len - limit);
        data.pred_close = data.pred_close.slice(data_len - limit);

        // predicted prices, 
        // becasue predicted result has one more day, so limit will not out of range
        data.next_close = new Array(limit)
        data.next_close.push(data.pred_close[limit])

        const high = Math.max(...[...data.real_close, ...data.pred_close])
        const low = Math.min(...[...data.real_close, ...data.pred_close])

        this.drawChart(data, 
        Math.ceil(high + (high - low)/5), 
        Math.floor(low - (high - low)/10),
        );
      }
    })
    .catch((e) => {
      console.info(e);
    });
  },


});
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
  section {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100vw;
    height: 100vh;

    .home {
      position: absolute;
      top: 30px;
      left: 50%;
      margin: 0 0 0 -2em;
    }

    .box {
      width: 1200px;
      height: 800px;
      margin-top: -50px;
      max-width: 80vw;
      max-height: 70vh;
    }

    color: rgb(233, 60, 88);
    color: rgb(46, 149, 98);
  }
</style>
