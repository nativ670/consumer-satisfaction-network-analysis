const nodesData = [
    {
        "id": "smell/fragrance",
        "pageRank": 0.1158
    },
    {
        "id": "price/value",
        "pageRank": 0.1089
    },
    {
        "id": "texture/consistency",
        "pageRank": 0.0915
    },
    {
        "id": "packaging",
        "pageRank": 0.239
    },
    {
        "id": "ingredients",
        "pageRank": 0.2499
    },
    {
        "id": "effectiveness/results",
        "pageRank": 0.0663
    },
    {
        "id": "service/shipping",
        "pageRank": 0.1285
    }
];
const linksData = [
    {
        "source": "smell/fragrance",
        "target": "texture/consistency",
        "value": 0.0281
    },
    {
        "source": "smell/fragrance",
        "target": "packaging",
        "value": 0.0243
    },
    {
        "source": "smell/fragrance",
        "target": "ingredients",
        "value": 0.0555
    },
    {
        "source": "price/value",
        "target": "packaging",
        "value": 0.0659
    },
    {
        "source": "price/value",
        "target": "service/shipping",
        "value": 0.0382
    },
    {
        "source": "texture/consistency",
        "target": "packaging",
        "value": 0.0224
    },
    {
        "source": "texture/consistency",
        "target": "ingredients",
        "value": 0.0304
    },
    {
        "source": "packaging",
        "target": "ingredients",
        "value": 0.0784
    },
    {
        "source": "packaging",
        "target": "service/shipping",
        "value": 0.0557
    },
    {
        "source": "ingredients",
        "target": "effectiveness/results",
        "value": 0.0525
    },
    {
        "source": "ingredients",
        "target": "service/shipping",
        "value": 0.0319
    }
];
