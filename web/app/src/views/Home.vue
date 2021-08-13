<template>
    <div class="core">
        <Greeting />

        <!-- <router-link :to="{ name: 'stock_list' }">
            <span>stocks</span>
        </router-link> -->

        <div class="nav">
            <router-link :to="{ name: 'stylize' }">
                <span>stylize</span>
            </router-link>
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
// import axios, {AxiosResponse} from "axios";
import Greeting from "@/components/Greeting.vue";

export default defineComponent({
    components: {
        Greeting,
    },
    data() {
        return {
            stockList: [],
        };
    },
    computed: {},
    // created() {

    // },
    methods: {},
});
</script>
<style lang="scss">
$main-color: #69ca62;

#app {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
    background: #005c97; /* fallback for old browsers */
}

.nav {
    position: absolute;
    top: 50%;
    left: 50%;
    margin: -50px 0 0 -200px;
    border: 1px solid $main-color;
    padding: 6px 10px;

    span {
        color: $main-color;
    }
}
</style>
